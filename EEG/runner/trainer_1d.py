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
        def normalize_1d(tensor):
            return (tensor - 0.5) / 0.5
        
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
            shuffle=True
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
        clean_signals = batch[0].to(self.config.device)
        
        # IMPROVED: Generate noise that matches signal characteristics
        signal_mean = clean_signals.mean()
        signal_std = clean_signals.std()

        noise = torch.randn(clean_signals.shape).to(clean_signals.device)
        noise = noise * (signal_std * 0.1) + signal_mean  # Much smaller noise scale
        
        # Sample a random timestep for each example
        timesteps = torch.randint(
            0, 
            self.noise_scheduler.config.num_train_timesteps,
            (clean_signals.shape[0],),
            device=clean_signals.device
        ).long()
        
        # FIXED: Add noise to the entire signal first, then condition
        noisy_signals = self.noise_scheduler.add_noise(clean_signals, noise, timesteps)
        
        # Apply conditioning after noise addition
        # Keep the conditioning region from the original clean signal
        noisy_signals[:, :, :self.config.prediction_point] = clean_signals[:, :, :self.config.prediction_point]
        
        # Predict the noise residual
        noise_pred = self.unet(noisy_signals, timesteps).sample
        
        # FIXED: Compute loss with proper weighting
        # Loss on full signal with higher weight on prediction region
        prediction_loss = F.mse_loss(
            noise_pred[:, :, self.config.prediction_point:],
            noise[:, :, self.config.prediction_point:]
        )
        
        # Optional: Add small loss on conditioning region to maintain consistency
        conditioning_loss = F.mse_loss(
            noise_pred[:, :, :self.config.prediction_point],
            noise[:, :, :self.config.prediction_point]
        )
        
        # Weighted combination - focus on prediction region
        loss = prediction_loss + 0.1 * conditioning_loss
        
        # FIXED: Gradient clipping before step
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        # Enhanced logging
        logs = {
            "epoch": (epoch // len(self.train_dataloader)),
            "iteration": epoch,
            "total_loss": loss.detach().item(),
            "prediction_loss": prediction_loss.detach().item(),
            "conditioning_loss": conditioning_loss.detach().item(),
            "lr": self.lr_scheduler.get_last_lr()[0]
        }
        
        # Print progress
        if epoch % 100 == 0 or epoch < 50:  # More frequent logging at start
            print(", ".join([key + ": " + str(round(value, 6)) for key, value in logs.items()]))
        
        wandb.log(logs)
        
        return loss.item()

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
            plt.ylim(-0.1, 1.1)
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
        
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            
            for iteration, batch in enumerate(self.train_dataloader):
                loss = self.train_single_batch(batch, number_iteration)
                epoch_losses.append(loss)
                number_iteration += 1
                
                # More frequent evaluation during early training
                eval_interval = self.config.eval_interval if number_iteration > 1000 else 200
                
                if (number_iteration >= self.config.get('eval_begin', 500) and 
                    number_iteration % eval_interval == 0):
                    
                    # FIXED: Evaluate on complete datasets, not just single batches
                    print(f"\n=== Full Dataset Evaluation at iteration {number_iteration} ===")
                    
                    # Evaluate full training dataset
                    train_data, train_mses = self.evaluate_full_dataset(
                        self.train_dataloader, "train", number_iteration, self.train_labels
                    )
                    
                    # Evaluate full validation dataset  
                    val_data, val_mses = self.evaluate_full_dataset(
                        self.val_dataloader, "test", number_iteration, self.test_labels
                    )
                    
                    # Create visualizations for both datasets
                    self.create_sample_visualizations(train_data, train_mses, "train", number_iteration, self.train_labels)
                    self.create_sample_visualizations(val_data, val_mses, "test", number_iteration, self.test_labels)
                    
                    # Use validation MSE for best model selection
                    val_avg_mse = np.mean(val_mses)
                    
                    # Save best model
                    if val_avg_mse < best_mse:
                        best_mse = val_avg_mse
                        torch.save(self.unet.state_dict(), 
                                 f"{self.config.output_dir}/best_model.pth")
                        print(f"New best model saved with validation MSE: {best_mse:.6f}")
                    
                    print(f"=== Evaluation Complete ===\n")
            
            # Log epoch statistics
            epoch_avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch}: Avg Loss = {epoch_avg_loss:.6f}")
            wandb.log({"epoch_avg_loss": epoch_avg_loss, "epoch": epoch})
        
        print(f"Training completed! Best validation MSE achieved: {best_mse:.6f}")
        
        # Final evaluation on complete datasets
        print("\n=== Final Complete Dataset Evaluation ===")
        final_train_data, final_train_mses = self.evaluate_full_dataset(
            self.train_dataloader, "train_final", "final", self.train_labels
        )
        final_val_data, final_val_mses = self.evaluate_full_dataset(
            self.val_dataloader, "test_final", "final", self.test_labels
        )
        
        # Create final visualizations
        self.create_sample_visualizations(final_train_data, final_train_mses, "train_final", "final", self.train_labels)
        self.create_sample_visualizations(final_val_data, final_val_mses, "test_final", "final", self.test_labels)
        
        print(f"Final Training Dataset: {final_train_data.shape}")
        print(f"Final Validation Dataset: {final_val_data.shape}")
        print(f"Final Training MSE: {np.mean(final_train_mses):.6f}")
        print(f"Final Validation MSE: {np.mean(final_val_mses):.6f}")
        print(f"Final Training Data Range: [{np.min(final_train_data):.4f}, {np.max(final_train_data):.4f}]")
        print(f"Final Validation Data Range: [{np.min(final_val_data):.4f}, {np.max(final_val_data):.4f}]")