import random
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn.functional as F
import pandas as pd
from collections import Counter
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
        
        # Move scheduler tensors to the correct device
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.config.device)
        self.noise_scheduler.alphas = self.noise_scheduler.alphas.to(self.config.device)
        self.noise_scheduler.betas = self.noise_scheduler.betas.to(self.config.device)
        if hasattr(self.noise_scheduler, 'final_alpha_cumprod'):
            self.noise_scheduler.final_alpha_cumprod = self.noise_scheduler.final_alpha_cumprod.to(self.config.device)
        
        self.pipeline = DDIMPipeline1D(unet=self.unet, scheduler=self.noise_scheduler)
        
        # Build datasets WITHOUT any additional transforms
        self.train_dataset = EEGDiffDR.build(train_dataset)
        self.val_dataset = EEGDiffDR.build(val_dataset)
        
        # Store original datasets for denormalization
        self.train_dataset_original = EEGDiffDR.build(train_dataset)
        self.val_dataset_original = EEGDiffDR.build(val_dataset)
        
        # NO ADDITIONAL TRANSFORMS - datasets already handle normalization
        
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
        
        # Optimizer
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

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = getattr(self.config, 'early_stopping_patience', 10)
        self.min_delta = getattr(self.config, 'min_delta', 0.001)
        
        self.train_losses = []
        self.val_losses = []
        
        # Load labels
        try:
            train_labels_path = getattr(self.config, 'train_labels_path', 'C:/Github/EEG-Mouse/data/train_labels.csv')
            test_labels_path = getattr(self.config, 'test_labels_path', 'C:/Github/EEG-Mouse/data/test_labels.csv')
            
            self.train_labels = pd.read_csv(train_labels_path, header=None).values.flatten()
            self.test_labels = pd.read_csv(test_labels_path, header=None).values.flatten()
            
            print(f"Loaded {len(self.train_labels)} training labels and {len(self.test_labels)} testing labels")
        except Exception as e:
            print(f"Error loading labels: {e}")
            self.train_labels = None
            self.test_labels = None

    def initial_unet(self):
        self.unet.to(self.config.device)
        if self.config.u_net_weight_path is not None:
            self.unet.load_state_dict(torch.load(self.config.u_net_weight_path, map_location=self.config.device))
            print(f"Load u_net weight from {self.config.u_net_weight_path}")
        else:
            print("No u_net weight path is provided, using random weights")

    def train_single_batch_fixed(self, batch, epoch):
        """
        Improved training that encourages pattern learning, not just denoising
        """
        clean_signals = batch[0].to(self.config.device)
        
        # Generate noise
        noise = torch.randn(clean_signals.shape, device=clean_signals.device)
        
        # Use lower timesteps more often to focus on detail preservation
        # This encourages learning patterns rather than just denoising
        if random.random() < 0.5:  # 50% of the time
            # Sample from lower timesteps (less noise)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps // 2,
                (clean_signals.shape[0],), device=clean_signals.device
            ).long()
        else:
            # Sample from full range
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (clean_signals.shape[0],), device=clean_signals.device
            ).long()
        
        # Add noise using scheduler
        noisy_signals = self.noise_scheduler.add_noise(clean_signals, noise, timesteps)
        
        # CRITICAL: Preserve conditioning exactly
        noisy_signals[:, :, :self.config.prediction_point] = clean_signals[:, :, :self.config.prediction_point].clone()
        
        # Target is the noise for the prediction region
        noise_target = noise.clone()
        noise_target[:, :, :self.config.prediction_point] = 0.0
        
        # UNet prediction
        noise_pred = self.unet(noisy_signals, timesteps).sample
        
        # ENSURE CONDITIONING IS NOT PREDICTED
        noise_pred[:, :, :self.config.prediction_point] = 0.0
        
        # Get alpha values
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(timesteps.device)
        alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1)
        beta_prod_t = (1 - alpha_prod_t).view(-1, 1, 1)
        
        # 1. Noise prediction loss (only on prediction region)
        noise_loss = F.mse_loss(
            noise_pred[:, :, self.config.prediction_point:], 
            noise_target[:, :, self.config.prediction_point:]
        )
        
        # 2. Direct signal prediction loss (more stable)
        # For low-noise samples, we can directly compare predicted vs true signal
        if timesteps.float().mean() < 0.3 * self.noise_scheduler.config.num_train_timesteps:
            pred_clean = (noisy_signals - beta_prod_t.sqrt() * noise_pred) / (alpha_prod_t.sqrt() + 1e-8)
            pred_clean = torch.clamp(pred_clean, -10, 10)
            
            signal_loss = F.mse_loss(
                pred_clean[:, :, self.config.prediction_point:],
                clean_signals[:, :, self.config.prediction_point:]
            )
        else:
            signal_loss = noise_loss.clone()
        
        # 3. IMPROVED Pattern Continuation Loss
        # This is the KEY to making the model learn patterns
        boundary_size = 64  # Look at last 64 points of conditioning
        if self.config.prediction_point > boundary_size:
            # Extract pattern from end of conditioning region
            conditioning_end = clean_signals[:, :, self.config.prediction_point-boundary_size:self.config.prediction_point]
            
            # Extract pattern from start of prediction region  
            prediction_start = clean_signals[:, :, self.config.prediction_point:self.config.prediction_point+boundary_size]
            
            # Compute pattern features (frequency, amplitude, etc)
            # Using FFT to capture frequency content
            cond_fft = torch.fft.rfft(conditioning_end, dim=-1)
            pred_fft = torch.fft.rfft(prediction_start, dim=-1)
            
            # Frequency consistency loss
            freq_loss = F.mse_loss(torch.abs(cond_fft), torch.abs(pred_fft))
            
            # Amplitude consistency
            cond_std = conditioning_end.std(dim=-1, keepdim=True)
            pred_std = prediction_start.std(dim=-1, keepdim=True)
            amp_loss = F.mse_loss(cond_std, pred_std)
            
            # Combined pattern loss
            pattern_loss = freq_loss + amp_loss
            
            # Also check predicted signal pattern matching
            if 'pred_clean' in locals():
                pred_clean_start = pred_clean[:, :, self.config.prediction_point:self.config.prediction_point+boundary_size]
                pred_clean_fft = torch.fft.rfft(pred_clean_start, dim=-1)
                pattern_pred_loss = F.mse_loss(torch.abs(pred_clean_fft), torch.abs(pred_fft))
                pattern_loss = pattern_loss + pattern_pred_loss
        else:
            pattern_loss = torch.tensor(0.0, device=clean_signals.device)
        
        # 4. Gradient continuity at boundary
        grad_window = 10
        if self.config.prediction_point > grad_window:
            # Gradients around boundary
            before_boundary = clean_signals[:, :, self.config.prediction_point-grad_window:self.config.prediction_point]
            after_boundary = clean_signals[:, :, self.config.prediction_point:self.config.prediction_point+grad_window]
            
            # Compute local gradients
            grad_before = before_boundary[:, :, 1:] - before_boundary[:, :, :-1]
            grad_after = after_boundary[:, :, 1:] - after_boundary[:, :, :-1]
            
            # Last gradient of before should match first gradient of after
            boundary_grad_loss = F.mse_loss(grad_before[:, :, -1:], grad_after[:, :, :1])
        else:
            boundary_grad_loss = torch.tensor(0.0, device=clean_signals.device)
        
        # ADJUSTED WEIGHTS for pattern learning
        total_loss = (
            0.4 * noise_loss +           # Still important for diffusion
            0.3 * signal_loss +          # Direct signal matching
            0.25 * pattern_loss +        # INCREASED for pattern learning
            0.05 * boundary_grad_loss    # Smooth transitions
        )
        
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        # Enhanced logging
        logs = {
            "epoch": (epoch // len(self.train_dataloader)),
            "iteration": epoch,
            "total_loss": total_loss.detach().item(),
            "noise_loss": noise_loss.detach().item(),
            "signal_loss": signal_loss.detach().item(),
            "pattern_loss": pattern_loss.detach().item(),
            "boundary_grad_loss": boundary_grad_loss.detach().item(),
            "grad_norm": grad_norm.item(),
            "lr": self.lr_scheduler.get_last_lr()[0],
            "mean_timestep": timesteps.float().mean().item(),
        }
        
        if epoch % 100 == 0:
            print(f"Step {epoch}: Total={total_loss:.4f}, Noise={noise_loss:.4f}, "
                f"Signal={signal_loss:.4f}, Pattern={pattern_loss:.4f}, "
                f"GradNorm={grad_norm:.3f}")
        
        wandb.log(logs)
        return total_loss.item()

    def visualize_predictions(self, original_data, predicted_data, sample_idx, epoch, dataset_name):
        """
        Visualize original vs predicted EEG signals
        """
        plt.figure(figsize=(15, 8))
        
        # Plot 1: Full signal comparison
        plt.subplot(3, 1, 1)
        plt.plot(original_data, label='Original', alpha=0.7, linewidth=1)
        plt.plot(predicted_data, label='Predicted', alpha=0.7, linewidth=1)
        plt.axvline(x=self.config.prediction_point, color='red', linestyle='--', label='Prediction Point')
        plt.title(f'Full Signal Comparison - Sample {sample_idx}')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Prediction region only
        plt.subplot(3, 1, 2)
        pred_start = self.config.prediction_point
        plt.plot(range(pred_start, len(original_data)), 
                original_data[pred_start:], label='Original', alpha=0.7, linewidth=1)
        plt.plot(range(pred_start, len(predicted_data)), 
                predicted_data[pred_start:], label='Predicted', alpha=0.7, linewidth=1)
        plt.title('Prediction Region Only')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Boundary region detail
        plt.subplot(3, 1, 3)
        boundary_start = max(0, self.config.prediction_point - 50)
        boundary_end = min(len(original_data), self.config.prediction_point + 100)
        plt.plot(range(boundary_start, boundary_end), 
                original_data[boundary_start:boundary_end], label='Original', alpha=0.7, linewidth=1)
        plt.plot(range(boundary_start, boundary_end), 
                predicted_data[boundary_start:boundary_end], label='Predicted', alpha=0.7, linewidth=1)
        plt.axvline(x=self.config.prediction_point, color='red', linestyle='--', label='Prediction Point')
        plt.title('Boundary Region Detail')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = f"{self.config.output_dir}/{epoch}/visualizations"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{dataset_name}_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also log to wandb
        wandb.log({f"{dataset_name}_visualization_sample_{sample_idx}": wandb.Image(plt)})

    def evaluate_full_dataset_fixed(self, dataloader, dataset_name, epoch, labels=None):
        """
        Fixed evaluation with proper z-score denormalization
        """
        print(f"Evaluating {dataset_name} dataset...")
        
        all_concatenated_data = []
        all_mses = []
        all_correlations = []
        
        # Get original dataset for denormalization
        if dataset_name.startswith("train"):
            original_dataset = self.train_dataset_original
        else:
            original_dataset = self.val_dataset_original
        
        total_samples = len(original_dataset.data)
        print(f"Processing {total_samples} samples from {dataset_name} dataset")
        
        processed_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                normalized_signal = batch[0].to(self.config.device)
                
                # Generate prediction
                result = self.pipeline(
                    normalized_signal,
                    self.config.prediction_point,
                    batch_size=len(normalized_signal),
                    num_inference_steps=self.config.num_inference_steps,
                )
                
                predicted_signal = result.images
                
                

                # Process each sample
                for i in range(len(normalized_signal)):
                    sample_idx = batch_idx * dataloader.batch_size + i
                    
                    if sample_idx >= total_samples:
                        continue
                    
                    # Get original data (before normalization)
                    original_data_full = original_dataset.data[sample_idx]
                    
                    # Get normalized prediction and denormalize
                    predicted_normalized = predicted_signal[i, 0].cpu().numpy()
                    
                    # Use dataset's denormalization method
                    predicted_denormalized = original_dataset.denormalize(predicted_normalized)
                    
                    # The full signal is already denormalized
                    full_signal_original_scale = predicted_denormalized
                    
                    all_concatenated_data.append(full_signal_original_scale)
                    
                    # Calculate metrics on denormalized data
                    original_second_half = original_data_full[self.config.prediction_point:]
                    predicted_second_half = predicted_denormalized[self.config.prediction_point:]
                    
                    mse = np.mean((predicted_second_half - original_second_half) ** 2)
                    all_mses.append(mse)
                    
                    # Calculate correlation
                    if len(original_second_half) > 1:
                        corr = np.corrcoef(original_second_half, predicted_second_half)[0, 1]
                        all_correlations.append(corr if not np.isnan(corr) else 0)
                    else:
                        all_correlations.append(0)
                    
                    # Visualize some samples
                    if sample_idx < 5 or sample_idx % 100 == 0:  # First 5 samples and every 100th
                        self.visualize_predictions(
                            original_data_full, 
                            predicted_denormalized, 
                            sample_idx, 
                            epoch, 
                            dataset_name
                        )

                    
                    processed_samples += 1
                
                if batch_idx % 15 == 0:
                    print(f"Processed batch {batch_idx+1}/{len(dataloader)}")
        
        print(f"Processed {processed_samples} samples")
        
        # Convert to arrays
        all_concatenated_data = np.array(all_concatenated_data)
        all_mses = np.array(all_mses)
        all_correlations = np.array(all_correlations)
        
        # Statistics
        avg_correlation = np.mean(all_correlations)
        positive_corr_count = np.sum(all_correlations > 0)
        strong_corr_count = np.sum(all_correlations > 0.3)
        
        print(f"Results for {dataset_name}:")
        print(f"  Average correlation: {avg_correlation:.6f}")
        print(f"  Positive correlations: {positive_corr_count} / {len(all_correlations)} ({100*positive_corr_count/len(all_correlations):.1f}%)")
        print(f"  Strong correlations (>0.3): {strong_corr_count} / {len(all_correlations)} ({100*strong_corr_count/len(all_correlations):.1f}%)")
        print(f"  Average MSE: {np.mean(all_mses):.6f}")
        
        # Save dataset
        father_path = f"{self.config.output_dir}/{epoch}"
        if not os.path.exists(father_path):
            os.makedirs(father_path)
        
        csv_filename = f"{father_path}/concatenated_{dataset_name}_data_epoch_{epoch}.csv"
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for sample in all_concatenated_data:
                writer.writerow(sample)
        
        print(f"Saved {dataset_name} data to {csv_filename}")
        
        # Log metrics
        wandb.log({
            f'{dataset_name}_mse': np.mean(all_mses),
            f'{dataset_name}_avg_correlation': avg_correlation,
            f'{dataset_name}_positive_correlations': positive_corr_count,
            f'{dataset_name}_strong_correlations': strong_corr_count,
        })
        
        # Visualize distribution of correlations
        self.visualize_correlation_distribution(all_correlations, dataset_name, epoch)
        
        return all_concatenated_data, all_mses, avg_correlation

    def visualize_correlation_distribution(self, correlations, dataset_name, epoch):
        """
        Visualize the distribution of correlations
        """
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(correlations, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='Zero correlation')
        plt.axvline(x=np.mean(correlations), color='green', linestyle='--', label=f'Mean: {np.mean(correlations):.3f}')
        plt.xlabel('Correlation')
        plt.ylabel('Count')
        plt.title(f'{dataset_name} - Correlation Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(correlations, vert=True)
        plt.ylabel('Correlation')
        plt.title(f'{dataset_name} - Correlation Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = f"{self.config.output_dir}/{epoch}/distributions"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{dataset_name}_correlation_dist.png", dpi=150, bbox_inches='tight')
        
        # Log to wandb
        wandb.log({f"{dataset_name}_correlation_distribution": wandb.Image(plt)})
        plt.close()

    def visualize_training_progress(self, epoch, train_losses, val_losses, train_correlations, val_correlations):
        """
        Visualize training progress over time
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot losses
        axes[0].plot(train_losses, label='Train Loss', alpha=0.7)
        axes[0].plot(val_losses, label='Val Loss', alpha=0.7)
        axes[0].set_xlabel('Evaluation Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot correlations
        axes[1].plot(train_correlations, label='Train Correlation', alpha=0.7)
        axes[1].plot(val_correlations, label='Val Correlation', alpha=0.7)
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Evaluation Step')
        axes[1].set_ylabel('Average Correlation')
        axes[1].set_title('Training and Validation Correlation Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = f"{self.config.output_dir}/progress"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/training_progress_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
        
        # Log to wandb
        wandb.log({"training_progress": wandb.Image(plt)})
        plt.close()

    def train(self):
        print("Starting training with z-score normalization...")
        print(f"Training dataset: {len(self.train_dataset)} samples")
        print(f"Validation dataset: {len(self.val_dataset)} samples")
        
        number_iteration = 0
        best_correlation = -1.0
        
        # Track metrics over time
        train_losses_history = []
        val_losses_history = []
        train_correlations_history = []
        val_correlations_history = []

        for epoch in range(self.config.num_epochs):
            epoch_losses = []

            for iteration, batch in enumerate(self.train_dataloader):
                # Use fixed training method
                training_loss = self.train_single_batch_fixed(batch, number_iteration)
                epoch_losses.append(training_loss)
                number_iteration += 1

                # Evaluation
                if number_iteration >= self.config.eval_begin and number_iteration % self.config.eval_interval == 0:
                    print(f"\n=== Evaluation at iteration {number_iteration} ===")

                    # Evaluate both datasets
                    train_data, train_mses, train_corr = self.evaluate_full_dataset_fixed(
                        self.train_dataloader, "train", number_iteration, self.train_labels
                    )

                    val_data, val_mses, val_corr = self.evaluate_full_dataset_fixed(
                        self.val_dataloader, "test", number_iteration, self.test_labels
                    )
                    
                    # Update history
                    train_losses_history.append(np.mean(epoch_losses))
                    val_losses_history.append(np.mean(val_mses))
                    train_correlations_history.append(train_corr)
                    val_correlations_history.append(val_corr)
                    
                    # Visualize progress
                    self.visualize_training_progress(
                        epoch, 
                        train_losses_history, 
                        val_losses_history,
                        train_correlations_history,
                        val_correlations_history
                    )

                    # Save best model based on validation correlation
                    if val_corr > best_correlation:
                        best_correlation = val_corr
                        torch.save(self.unet.state_dict(), 
                                 f"{self.config.output_dir}/best_model_zscore.pth")
                        print(f"New best model saved with correlation: {best_correlation:.6f}")
                        
                        # Save checkpoint with full state
                        checkpoint = {
                            'epoch': epoch,
                            'iteration': number_iteration,
                            'model_state_dict': self.unet.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.lr_scheduler.state_dict(),
                            'best_correlation': best_correlation,
                            'train_mean': self.train_dataset.mean,
                            'train_std': self.train_dataset.std,
                        }
                        torch.save(checkpoint, f"{self.config.output_dir}/best_checkpoint_zscore.pth")

                    print(f"=== Evaluation Complete ===\n")

            # Epoch summary
            epoch_avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch}: Avg Training Loss = {epoch_avg_loss:.6f}")

        print(f"Training completed! Best correlation: {best_correlation:.6f}")

        # Final evaluation
        print("\n=== Final Evaluation ===")
        final_train_data, _, _ = self.evaluate_full_dataset_fixed(
           self.train_dataloader, "train_final", "final", self.train_labels
        )
        final_val_data, _, _ = self.evaluate_full_dataset_fixed(
           self.val_dataloader, "test_final", "final", self.test_labels
        )

        print(f"Training complete!")
        print(f"Best model saved at: {self.config.output_dir}/best_model_zscore.pth")

        wandb.log({
            "final_best_correlation": best_correlation,
            "training_completed": True
        })