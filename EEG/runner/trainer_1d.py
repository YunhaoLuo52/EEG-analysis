import random
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from diffusers.optimization import get_cosine_schedule_with_warmup
from mmengine import Config
from ..registry import EEGDiffMR, EEGDiffDR
from ..pipeline import DDIMPipeline1D

@EEGDiffMR.register_module()
class EEGDiffTrainner1D:
    def __init__(self,
                 trainner_config: Config,
                 unet: Config,
                 noise_scheduler: Config,
                 optimizer: Config,
                 train_dataset: Config,
                 val_dataset: Config):
        self.config = trainner_config
        self.unet = EEGDiffMR.build(unet)
        self.initial_unet()
        self.noise_scheduler = EEGDiffMR.build(noise_scheduler)
        self.pipeline = DDIMPipeline1D(unet=self.unet, scheduler=self.noise_scheduler)
        
        # Define normalization function
        # def normalize_1d(tensor):
        #     return (tensor - 0.5) / 0.5
        def normalize_1d(tensor):
            # Use z-score normalization instead
            mean = tensor.mean()
            std = tensor.std()
            return (tensor - mean) / (std + 1e-8)

        self.train_dataset = EEGDiffDR.build(train_dataset)
        self.val_dataset = EEGDiffDR.build(val_dataset)
        self.train_dataset.transform = normalize_1d
        self.val_dataset.transform = normalize_1d
        
        # Don't apply additional transform - use data as-is from dataset
        # The dataset already normalizes to [0,1] which is appropriate
        
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.train_batch_size, 
            shuffle=True
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size=self.config.eval_batch_size, 
            shuffle=False
        )
        
        # FIXED: Use AdamW with better parameters
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(), 
            lr=optimizer.learning_rate,
            weight_decay=optimizer.get('weight_decay', 0.01),
            betas=optimizer.get('betas', (0.9, 0.999)),
            eps=optimizer.get('eps', 1e-8)
        )
        
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=(len(self.train_dataloader) * self.config.num_epochs),
        )

         # EARLY STOPPING VARIABLES
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = getattr(self.config, 'early_stopping_patience', 10)
        self.min_delta = getattr(self.config, 'min_delta', 0.001)
        
        # VALIDATION LOSS TRACKING
        self.train_losses = []
        self.val_losses = []
        
        # Load labels for seizure/non-seizure classification
        try:
            train_labels_path = getattr(self.config, 'train_labels_path', 'C:/Github/EEG-Mouse/data/train_labels.csv')
            test_labels_path = getattr(self.config, 'test_labels_path', 'C:/Github/EEG-Mouse/data/test_labels.csv')
            
            self.train_labels = pd.read_csv(train_labels_path, header=None).values.flatten()
            self.test_labels = pd.read_csv(test_labels_path, header=None).values.flatten()
            
            print(f"Loaded {len(self.train_labels)} training labels and {len(self.test_labels)} testing labels")
        except Exception as e:
            print(f"Error loading labels: {e}")
            print("Will continue without seizure/non-seizure classification")
            self.train_labels = None
            self.test_labels = None

    def initial_unet(self):
        self.unet.to(self.config.device)
        if self.config.u_net_weight_path is not None:
            self.unet.load_state_dict(torch.load(self.config.u_net_weight_path, map_location=self.config.device))
            print(f"Load u_net weight from {self.config.u_net_weight_path}")
        else:
            print("No u_net weight path is provided, using random weights")

    def validate_full_dataset(self):
        """
        Validate on the entire validation dataset to get reliable metrics
        """
        self.unet.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                clean_signals = batch[0].to(self.config.device)
                
                # Generate noise
                noise = torch.randn(clean_signals.shape).to(clean_signals.device)
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, 
                    self.noise_scheduler.config.num_train_timesteps,
                    (clean_signals.shape[0],),
                    device=clean_signals.device
                ).long()
                
                # Add noise to signals
                signals_with_noise = self.noise_scheduler.add_noise(clean_signals, noise, timesteps)
                
                # Keep conditioning part unchanged
                signals_with_noise[:, :, :self.config.prediction_point] = clean_signals[:, :, :self.config.prediction_point]
                
                # Predict noise
                noise_pred = self.unet(signals_with_noise, timesteps).sample
                
                # Calculate loss only on predicted part
                loss = F.mse_loss(
                    noise_pred[:, :, self.config.prediction_point:],
                    noise[:, :, self.config.prediction_point:]
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        self.unet.train()
        avg_val_loss = total_loss / num_batches
        return avg_val_loss

    def should_stop_early(self, val_loss):
        """
        Check if training should stop early based on validation loss
        """
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered! No improvement for {self.early_stopping_patience} evaluations")
                return True
            return False
        
    def train_single_batch(self, batch, epoch):
        """FIXED: Train UNet to properly respect conditioning"""
        clean_signals = batch[0].to(self.config.device)
        
        # Generate noise for entire signal
        noise = torch.randn(clean_signals.shape, device=clean_signals.device)
        
        # CRITICAL FIX: Create proper noise target
        # Conditioning region should have ZERO noise target
        noise_target = noise.clone()
        noise_target[:, :, :self.config.prediction_point] = 0.0  # Zero for conditioning
        
        # Sample timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (clean_signals.shape[0],), device=clean_signals.device
        ).long()
        
        # Add noise to entire signal
        noisy_signals = self.noise_scheduler.add_noise(clean_signals, noise, timesteps)
        
        # Apply conditioning - keep conditioning region clean
        noisy_signals[:, :, :self.config.prediction_point] = clean_signals[:, :, :self.config.prediction_point]
        
        # UNet prediction
        noise_pred = self.unet(noisy_signals, timesteps).sample
        
        # FIXED LOSS: Train on entire signal with proper targets
        loss = F.mse_loss(noise_pred, noise_target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        # Enhanced logging
        logs = {
            "epoch": (epoch // len(self.train_dataloader)),
            "iteration": epoch,
            "training_loss": loss.detach().item(),
            "lr": self.lr_scheduler.get_last_lr()[0]
        }
        
        # Print progress
        if epoch % 100 == 0 or epoch < 50:
            print(", ".join([key + ": " + str(round(value, 6)) for key, value in logs.items()]))
        
        wandb.log(logs)
        
        return loss.item()


    # This preserves EEG characteristics better than simple MSE
    def train_single_batch_advanced_loss(self, batch, epoch):
        """Advanced training with EEG-specific loss for classification quality"""
        clean_signals = batch[0].to(self.config.device)
        
        # Same noise setup as before
        noise = torch.randn(clean_signals.shape, device=clean_signals.device)
        noise_target = noise.clone()
        noise_target[:, :, :self.config.prediction_point] = 0.0
        
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                (clean_signals.shape[0],), device=clean_signals.device).long()
        
        noisy_signals = self.noise_scheduler.add_noise(clean_signals, noise, timesteps)
        noisy_signals[:, :, :self.config.prediction_point] = clean_signals[:, :, :self.config.prediction_point]
        
        noise_pred = self.unet(noisy_signals, timesteps).sample
        
        # ADVANCED LOSS COMPUTATION
        # 1. Base MSE loss (noise prediction)
        mse_loss = F.mse_loss(noise_pred, noise_target)
        
        # 2. Frequency preservation loss
        # Predict clean signal from noise prediction
        predicted_clean = noisy_signals - noise_pred
        target_clean = clean_signals
        
        # Focus on prediction region for frequency loss
        pred_region = predicted_clean[:, :, self.config.prediction_point:]
        target_region = target_clean[:, :, self.config.prediction_point:]
        
        # FFT-based frequency loss
        pred_fft = torch.fft.fft(pred_region, dim=-1)
        target_fft = torch.fft.fft(target_region, dim=-1)
        freq_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
        
        # 3. Gradient loss (preserves signal morphology)
        pred_grad = pred_region[:, :, 1:] - pred_region[:, :, :-1]
        target_grad = target_region[:, :, 1:] - target_region[:, :, :-1]
        grad_loss = F.mse_loss(pred_grad, target_grad)
        
        # 4. Amplitude preservation loss
        pred_std = torch.std(pred_region, dim=-1, keepdim=True)
        target_std = torch.std(target_region, dim=-1, keepdim=True)
        amplitude_loss = F.mse_loss(pred_std, target_std)
        
        # Combined loss with weights
        total_loss = (mse_loss + 
                    0.3 * freq_loss + 
                    0.2 * grad_loss + 
                    0.1 * amplitude_loss)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=0.5)  # Lower clip for stability
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        # Enhanced logging
        logs = {
            "epoch": (epoch // len(self.train_dataloader)),
            "iteration": epoch,
            "total_loss": total_loss.detach().item(),
            "mse_loss": mse_loss.detach().item(),
            "freq_loss": freq_loss.detach().item(),
            "grad_loss": grad_loss.detach().item(),
            "amplitude_loss": amplitude_loss.detach().item(),
            "lr": self.lr_scheduler.get_last_lr()[0]
        }
        
        if epoch % 100 == 0:
            print(f"Step {epoch}: Total={total_loss:.6f}, MSE={mse_loss:.6f}, Freq={freq_loss:.6f}")
        
        wandb.log(logs)
        return total_loss.item()



    def create_sample_visualizations(self, concatenated_data_norm, mses, dataset_name, epoch, labels=None):
        """Create sample visualizations for the evaluation using denormalized data"""
        father_path = f"{self.config.output_dir}/{epoch}"
        samples_dir = os.path.join(father_path, f"{dataset_name}_samples")
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)
        
        # Select samples for visualization
        num_samples = min(10, len(concatenated_data_norm))
        sorted_indices = np.argsort(mses)
        step = max(1, len(sorted_indices) // num_samples)
        selected_indices = sorted_indices[::step][:num_samples]
        
        # Create individual plots for selected samples
        for i, idx in enumerate(selected_indices):
            plt.figure(figsize=(15, 5))
            
            # Plot full signal with prediction point marked (using denormalized data)
            plt.plot(concatenated_data_norm[idx], label='Concatenated (Original + Predicted)', linewidth=1)
            plt.axvline(x=self.config.prediction_point, color='r', linestyle='--', label='Prediction Point')
            
            # Add label if available
            label_str = ""
            if labels is not None and idx < len(labels):
                label_str = " (Seizure)" if labels[idx] == 1 else " (Non-Seizure)"
            
            plt.title(f'{dataset_name} Sample {i+1}: MSE={mses[idx]:.4f}{label_str}')
            plt.xlabel('Time Points')
            plt.ylabel('EEG Signal (Normalized [0,1])')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{samples_dir}/sample_{i+1}_concatenated.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # Create MSE distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(mses, bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'{dataset_name} MSE Distribution (Avg: {np.mean(mses):.4f})')
        plt.xlabel('MSE')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{samples_dir}/mse_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create data range comparison plot
        plt.figure(figsize=(12, 8))
        
        # Plot first few samples to show data range
        num_range_samples = min(5, len(concatenated_data_norm))
        for i in range(num_range_samples):
            plt.subplot(num_range_samples, 1, i+1)
            plt.plot(concatenated_data_norm[selected_indices[i]], linewidth=1)
            plt.axvline(x=self.config.prediction_point, color='r', linestyle='--', alpha=0.7)
            plt.title(f'Sample {i+1} - Range: [{np.min(concatenated_data_norm[selected_indices[i]]):.2f}, {np.max(concatenated_data_norm[selected_indices[i]]):.2f}]')
            plt.ylabel('EEG Signal (Normalized)')  # CHANGED: Updated label
            if i == num_range_samples - 1:
                plt.xlabel('Time Points')
        
        plt.tight_layout()
        plt.savefig(f"{samples_dir}/data_range_samples.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Created visualizations for {dataset_name} dataset in {samples_dir}")
        print(f"Normalized data range: [{np.min(concatenated_data_norm):.4f}, {np.max(concatenated_data_norm):.4f}]")  # CHANGED: Updated message
    
    
    def evaluate_full_dataset(self, dataloader, dataset_name, epoch, labels=None):
        """
        Evaluate on the complete dataset and save concatenated data with correct shape
        FIXED: Now denormalizes data back to original scale before saving
        """
        print(f"Evaluating full {dataset_name} dataset...")
        
        all_concatenated_data_normalized = []
        all_concatenated_data_denormalized = []
        all_mses = []
        
        # Get the dataset object to access denormalization parameters
        dataset_obj = dataloader.dataset
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                original_signal = batch[0].to(self.config.device)
                
                # Generate predicted signal for this batch
                result = self.pipeline(
                    original_signal,
                    self.config.prediction_point,
                    batch_size=len(original_signal),
                    num_inference_steps=50,
                )
                
                predicted_signal = result.images
                
                # Process each sample in the batch
                for i in range(len(original_signal)):
                    # Get original first half (before prediction point) - normalized
                    first_half_norm = original_signal[i, 0, :self.config.prediction_point].cpu().numpy()
                    # Get predicted second half (after prediction point) - normalized
                    second_half_norm = predicted_signal[i, 0, self.config.prediction_point:].cpu().numpy()
                    # Concatenate normalized data
                    full_signal_norm = np.concatenate([first_half_norm, second_half_norm])
                    all_concatenated_data_normalized.append(full_signal_norm)
                    
                    # FIXED: Denormalize the full concatenated signal back to original scale
                    full_signal_denorm = dataset_obj.denormalize_with_min_max(full_signal_norm)
                    all_concatenated_data_denormalized.append(full_signal_denorm)
                    
                    # Calculate MSE for this sample (using normalized data for consistency)
                    original_second_half = original_signal[i, 0, self.config.prediction_point:].cpu().numpy()
                    mse = np.mean((second_half_norm - original_second_half) ** 2)
                    all_mses.append(mse)
                
                print(f"Processed batch {batch_idx+1}/{len(dataloader)} of {dataset_name} dataset")
        
        # Convert to numpy arrays
        all_concatenated_data_normalized = np.array(all_concatenated_data_normalized)
        all_concatenated_data_denormalized = np.array(all_concatenated_data_denormalized)
        all_mses = np.array(all_mses)
        
        # Create output directory
        father_path = f"{self.config.output_dir}/{epoch}"
        if not os.path.exists(father_path):
            os.makedirs(father_path)
        
        # FIXED: Save denormalized (original scale) concatenated dataset as CSV
        csv_filename = f"{father_path}/concatenated_{dataset_name}_data_epoch_{epoch}.csv"
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for sample in all_concatenated_data_denormalized:
                writer.writerow(sample)
        
        print(f"Saved DENORMALIZED concatenated {dataset_name} data to {csv_filename} with shape: {all_concatenated_data_denormalized.shape}")
        print(f"Data range: {np.min(all_concatenated_data_denormalized):.4f} to {np.max(all_concatenated_data_denormalized):.4f}")
        
        # OPTIONAL: Also save normalized version for debugging
        csv_filename_norm = f"{father_path}/concatenated_{dataset_name}_data_normalized_epoch_{epoch}.csv"
        with open(csv_filename_norm, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for sample in all_concatenated_data_normalized:
                writer.writerow(sample)
        
        print(f"Also saved normalized version to {csv_filename_norm}")
        
        # Log statistics
        avg_mse = np.mean(all_mses)
        wandb.log({
            f'{dataset_name}_full_dataset_mse': avg_mse,
            f'{dataset_name}_dataset_size': len(all_concatenated_data_denormalized),
            f'{dataset_name}_data_range_min': float(np.min(all_concatenated_data_denormalized)),
            f'{dataset_name}_data_range_max': float(np.max(all_concatenated_data_denormalized))
        })
        
        print(f"{dataset_name.capitalize()} dataset - Samples: {len(all_concatenated_data_denormalized)}, Avg MSE: {avg_mse:.6f}")
        
        return all_concatenated_data_normalized, all_mses

    def train(self):
        print("Starting enhanced training with fixes...")
        print(f"Data range: {torch.min(next(iter(self.train_dataloader))[0]):.4f} to {torch.max(next(iter(self.train_dataloader))[0]):.4f}")
    
        # Print dataset sizes for verification
        print(f"Training dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Training batches: {len(self.train_dataloader)}")
        print(f"Validation batches: {len(self.val_dataloader)}")

        number_iteration = 0
        best_mse = float('inf')

        # FIXED: Track both training losses and evaluation MSEs separately
        training_losses_history = []
        eval_mse_history = []

        for epoch in range(self.config.num_epochs):
            epoch_losses = []

            for iteration, batch in enumerate(self.train_dataloader):
                # Get training loss (noise prediction loss)
                # training_loss = self.train_single_batch(batch, number_iteration)
                training_loss = self.train_single_batch_advanced_loss(batch, number_iteration)
                epoch_losses.append(training_loss)
                training_losses_history.append(training_loss)
                number_iteration += 1

                # More frequent evaluation during early training
                eval_interval = self.config.eval_interval if number_iteration > 1000 else 200

                if (number_iteration >= self.config.get('eval_begin', 500) and 
                    number_iteration % eval_interval == 0):

                    print(f"\n=== Full Dataset Evaluation at iteration {number_iteration} ===")

                    # PRESERVE: Your original visualization workflow
                    # Evaluate full training dataset
                    train_data, train_mses = self.evaluate_full_dataset(
                        self.train_dataloader, "train", number_iteration, self.train_labels
                    )

                    # Evaluate full validation dataset  
                    val_data, val_mses = self.evaluate_full_dataset(
                        self.val_dataloader, "test", number_iteration, self.test_labels
                    )

                    # PRESERVE: Create visualizations for both datasets
                    self.create_sample_visualizations(train_data, train_mses, "train", number_iteration, self.train_labels)
                    self.create_sample_visualizations(val_data, val_mses, "test", number_iteration, self.test_labels)

                    # FIXED: Calculate and track the metrics correctly
                    train_signal_mse = np.mean(train_mses)
                    val_signal_mse = np.mean(val_mses)
                    recent_training_loss = np.mean(epoch_losses[-10:]) if len(epoch_losses) >= 10 else np.mean(epoch_losses)

                    # Store evaluation MSE history for trend analysis
                    eval_mse_history.append(val_signal_mse)

                    # ENHANCED: Log both training loss and evaluation MSE separately
                    wandb.log({
                        # Training metrics (noise prediction)
                        "training_noise_loss": recent_training_loss,
                        "training_loss_trend": np.mean(training_losses_history[-100:]) if len(training_losses_history) >= 100 else  recent_training_loss,

                        # Evaluation metrics (signal completion - what we actually care about)
                        "train_signal_mse": train_signal_mse,
                        "val_signal_mse": val_signal_mse,
                        "signal_mse_gap": val_signal_mse - train_signal_mse,

                        # Iteration tracking
                        "iteration": number_iteration,
                        "epoch_progress": epoch + (iteration / len(self.train_dataloader))
                    })

                    # FIXED: Early stopping based on signal MSE, not training loss
                    if len(eval_mse_history) >= 3:
                        # Check trend over last 3 evaluations
                        recent_mse_trend = np.mean(eval_mse_history[-2:]) - np.mean(eval_mse_history[-3:-1])

                        if recent_mse_trend > self.config.get('min_delta', 0.001):
                            self.patience_counter += 1
                            print(f"⚠️  Signal MSE not improving (trend: +{recent_mse_trend:.6f}). Patience: {self. patience_counter}/{self.early_stopping_patience}")
                        else:
                            self.patience_counter = 0
                            print(f"✅ Signal MSE improved (trend: {recent_mse_trend:.6f})")

                        # Early stopping check
                        if self.patience_counter >= self.early_stopping_patience:
                            print(f"🛑 Early stopping triggered! Signal MSE hasn't improved for {self.early_stopping_patience}  evaluations")
                            print(f"Best validation MSE achieved: {best_mse:.6f}")
                            return  # Exit training early

                    # PRESERVE: Save best model based on validation signal MSE
                    if val_signal_mse < best_mse:
                        best_mse = val_signal_mse
                        torch.save(self.unet.state_dict(), 
                                 f"{self.config.output_dir}/best_model.pth")
                        print(f"🎯 New best model saved with validation Signal MSE: {best_mse:.6f}")

                    # ENHANCED: Print comprehensive progress
                    print(f"📊 Metrics Summary:")
                    print(f"   Training Loss (noise): {recent_training_loss:.6f}")
                    print(f"   Train Signal MSE: {train_signal_mse:.6f}")
                    print(f"   Val Signal MSE: {val_signal_mse:.6f}")
                    print(f"   MSE Gap: {val_signal_mse - train_signal_mse:.6f}")
                    print(f"   Best Val MSE: {best_mse:.6f}")
                    print(f"=== Evaluation Complete ===\n")

            # Log epoch statistics
            epoch_avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch}: Avg Training Loss = {epoch_avg_loss:.6f}")
            wandb.log({"epoch_avg_training_loss": epoch_avg_loss, "epoch": epoch})

        print(f"Training completed! Best validation Signal MSE achieved: {best_mse:.6f}")

        # PRESERVE: Final evaluation on complete datasets with visualizations
        print("\n=== Final Complete Dataset Evaluation ===")
        final_train_data, final_train_mses = self.evaluate_full_dataset(
           self.train_dataloader, "train_final", "final", self.train_labels
        )
        final_val_data, final_val_mses = self.evaluate_full_dataset(
           self.val_dataloader, "test_final", "final", self.test_labels
        )

        # PRESERVE: Create final visualizations
        self.create_sample_visualizations(final_train_data, final_train_mses, "train_final", "final", self.train_labels)
        self.create_sample_visualizations(final_val_data, final_val_mses, "test_final", "final", self.test_labels)
    
        # ENHANCED: Final comprehensive summary
        final_train_mse = np.mean(final_train_mses)
        final_val_mse = np.mean(final_val_mses)

        print(f"📈 Final Results Summary:")
        print(f"   Final Training Dataset: {final_train_data.shape}")
        print(f"   Final Validation Dataset: {final_val_data.shape}")
        print(f"   Final Training Signal MSE: {final_train_mse:.6f}")
        print(f"   Final Validation Signal MSE: {final_val_mse:.6f}")
        print(f"   Final MSE Gap: {final_val_mse - final_train_mse:.6f}")
        print(f"   Best Validation MSE During Training: {best_mse:.6f}")
        print(f"   Final Training Data Range: [{np.min(final_train_data):.4f}, {np.max(final_train_data):.4f}]")
        print(f"   Final Validation Data Range: [{np.min(final_val_data):.4f}, {np.max(final_val_data):.4f}]")

        # Log final metrics
        wandb.log({
            "final_train_signal_mse": final_train_mse,
            "final_val_signal_mse": final_val_mse,
            "final_mse_gap": final_val_mse - final_train_mse,
            "best_val_mse_achieved": best_mse,
            "training_completed": True
        })



