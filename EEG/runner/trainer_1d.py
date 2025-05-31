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
        self.pipeline = DDIMPipeline1D(unet=self.unet, scheduler=self.noise_scheduler)
        
        # PATTERN-FOCUSED: Normalization optimized for pattern preservation
        def normalize_for_patterns(tensor):
            # Use z-score normalization but preserve more signal structure
            mean = tensor.mean(dim=-1, keepdim=True)
            std = tensor.std(dim=-1, keepdim=True)
            normalized = (tensor - mean) / (std + 1e-8)
            # Use larger scaling to preserve pattern details
            normalized = normalized * 0.7  # Increased from 0.5 to preserve more patterns
            return normalized

        self.train_dataset = EEGDiffDR.build(train_dataset)
        self.val_dataset = EEGDiffDR.build(val_dataset)
        
        # Store original datasets
        self.train_dataset_original = EEGDiffDR.build(train_dataset)
        self.val_dataset_original = EEGDiffDR.build(val_dataset)
        
        # Store normalization parameters
        self.normalization_scaling = 0.7  # Updated scaling factor
        
        # Apply pattern-focused normalization
        self.train_dataset.transform = normalize_for_patterns
        self.val_dataset.transform = normalize_for_patterns
        
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
        
        # PATTERN-FOCUSED: Lower learning rate for more careful pattern learning
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(), 
            lr=optimizer.learning_rate * 0.7,  # Reduce LR for more careful learning
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

    def train_single_batch_pattern_focused(self, batch, epoch):
        """
        PATTERN-FOCUSED: Training specifically designed to fix negative correlation issue
        """
        clean_signals = batch[0].to(self.config.device)
        
        # PATTERN-FOCUSED: More moderate noise to preserve pattern learning
        noise = torch.randn(clean_signals.shape, device=clean_signals.device)
        
        # Reduce noise level to focus on pattern learning
        noise = noise * 0.25  # Reduced from 0.35 to 0.25 for better pattern preservation
        
        # Proper noise target
        noise_target = noise.clone()
        noise_target[:, :, :self.config.prediction_point] = 0.0
        
        # Use lower timesteps to focus on pattern learning, not just noise removal
        max_timestep = int(self.noise_scheduler.config.num_train_timesteps * 0.6)  # Reduced from 0.85 to 0.6
        timesteps = torch.randint(
            0, max_timestep,
            (clean_signals.shape[0],), device=clean_signals.device
        ).long()
        
        # Add noise to signals
        noisy_signals = self.noise_scheduler.add_noise(clean_signals, noise, timesteps)
        noisy_signals[:, :, :self.config.prediction_point] = clean_signals[:, :, :self.config.prediction_point]
        
        # UNet prediction
        noise_pred = self.unet(noisy_signals, timesteps).sample
        
        # PATTERN-FOCUSED LOSS: Completely redesigned to fix negative correlation
        # 1. Basic noise prediction loss (very low weight)
        noise_loss = F.mse_loss(noise_pred, noise_target)
        
        # 2. Direct signal reconstruction loss (HIGHEST priority)
        predicted_clean = noisy_signals - noise_pred
        signal_loss = F.mse_loss(
            predicted_clean[:, :, self.config.prediction_point:],
            clean_signals[:, :, self.config.prediction_point:]
        )
        
        # 3. CRITICAL: Correlation-enhancing loss
        pred_region = predicted_clean[:, :, self.config.prediction_point:]
        target_region = clean_signals[:, :, self.config.prediction_point:]
        
        if pred_region.shape[-1] > 3:
            # Calculate correlation and encourage positive correlation
            pred_flat = pred_region.flatten()
            target_flat = target_region.flatten()
            
            # Normalize for correlation calculation
            pred_normalized = (pred_flat - pred_flat.mean()) / (pred_flat.std() + 1e-8)
            target_normalized = (target_flat - target_flat.mean()) / (target_flat.std() + 1e-8)
            
            # Correlation loss: encourage positive correlation
            correlation_loss = 1.0 - torch.mean(pred_normalized * target_normalized)
        else:
            correlation_loss = torch.tensor(0.0, device=clean_signals.device)
        
        # 4. PATTERN PRESERVATION: Local pattern matching
        if pred_region.shape[-1] > 5:
            # Compare local windows of the signals
            window_size = 5
            pattern_loss = 0.0
            num_windows = pred_region.shape[-1] - window_size + 1
            
            for i in range(0, num_windows, 2):  # Every other window to avoid overfitting
                pred_window = pred_region[:, :, i:i+window_size]
                target_window = target_region[:, :, i:i+window_size]
                
                # Normalize windows
                pred_win_norm = (pred_window - pred_window.mean()) / (pred_window.std() + 1e-8)
                target_win_norm = (target_window - target_window.mean()) / (target_window.std() + 1e-8)
                
                pattern_loss += F.mse_loss(pred_win_norm, target_win_norm)
            
            pattern_loss = pattern_loss / (num_windows // 2)
        else:
            pattern_loss = torch.tensor(0.0, device=clean_signals.device)
        
        # 5. Continuity at boundary (encourage smooth connection)
        if self.config.prediction_point > 2 and self.config.prediction_point < clean_signals.shape[-1] - 2:
            # Look at values around the boundary
            boundary_original = clean_signals[:, :, self.config.prediction_point-2:self.config.prediction_point+2]
            boundary_predicted = predicted_clean[:, :, self.config.prediction_point-2:self.config.prediction_point+2]
            
            # Encourage similar trends across boundary
            orig_trend = boundary_original[:, :, 2:] - boundary_original[:, :, :-2]
            pred_trend = boundary_predicted[:, :, 2:] - boundary_predicted[:, :, :-2]
            
            boundary_loss = F.mse_loss(pred_trend, orig_trend)
        else:
            boundary_loss = torch.tensor(0.0, device=clean_signals.device)
        
        # 6. Statistical matching (preserve signal characteristics)
        pred_mean = torch.mean(pred_region, dim=-1, keepdim=True)
        target_mean = torch.mean(target_region, dim=-1, keepdim=True)
        mean_loss = F.mse_loss(pred_mean, target_mean)
        
        pred_std = torch.std(pred_region, dim=-1, keepdim=True)
        target_std = torch.std(target_region, dim=-1, keepdim=True)
        std_loss = F.mse_loss(pred_std, target_std)
        
        # PATTERN-FOCUSED loss weights: Prioritize correlation and pattern matching
        total_loss = (
            0.1 * noise_loss +           # Minimal diffusion weight
            0.3 * signal_loss +          # Direct reconstruction
            0.25 * correlation_loss +    # CRITICAL: Fix negative correlation
            0.2 * pattern_loss +         # Pattern preservation
            0.1 * boundary_loss +        # Boundary continuity
            0.03 * mean_loss +           # Statistical matching
            0.02 * std_loss              # Std preservation
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)  # Lower clip for stability
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        # Enhanced logging
        logs = {
            "epoch": (epoch // len(self.train_dataloader)),
            "iteration": epoch,
            "total_loss": total_loss.detach().item(),
            "signal_loss": signal_loss.detach().item(),
            "correlation_loss": correlation_loss.detach().item(),
            "pattern_loss": pattern_loss.detach().item(),
            "boundary_loss": boundary_loss.detach().item(),
            "mean_loss": mean_loss.detach().item(),
            "std_loss": std_loss.detach().item(),
            "lr": self.lr_scheduler.get_last_lr()[0]
        }
        
        if epoch % 100 == 0:
            print(f"Step {epoch}: Total={total_loss:.6f}, Signal={signal_loss:.6f}, Corr={correlation_loss:.6f}, Pattern={pattern_loss:.6f}")
        
        wandb.log(logs)
        return total_loss.item()

    def fixed_denormalize(self, normalized_predictions, original_conditioning_data):
        """
        IMPROVED: Fixed denormalization with updated scaling factor
        """
        conditioning_mean = np.mean(original_conditioning_data)
        conditioning_std = np.std(original_conditioning_data)
        
        # Method 1: Standard denormalization with updated scaling
        method1 = (normalized_predictions / self.normalization_scaling) * conditioning_std + conditioning_mean
        
        # Method 2: Range-based with updated scaling
        conditioning_range = np.max(original_conditioning_data) - np.min(original_conditioning_data)
        normalized_range = np.max(normalized_predictions) - np.min(normalized_predictions)
        
        if normalized_range > 0:
            normalized_centered = normalized_predictions - np.mean(normalized_predictions)
            scale_factor = conditioning_std / (normalized_range / 3.0)  # Updated scaling
            method2 = normalized_centered * scale_factor + conditioning_mean
        else:
            method2 = np.full_like(normalized_predictions, conditioning_mean)
            
        # Method 3: Direct range mapping with updated scaling
        if np.max(normalized_predictions) != np.min(normalized_predictions):
            norm_min = np.min(normalized_predictions)
            norm_max = np.max(normalized_predictions)
            
            scale_factor = conditioning_range / (norm_max - norm_min)
            method3 = (normalized_predictions - norm_min) * scale_factor + np.min(original_conditioning_data)
        else:
            method3 = np.full_like(normalized_predictions, conditioning_mean)
        
        # Choose method that gives best std ratio
        method1_std = np.std(method1)
        method2_std = np.std(method2)
        method3_std = np.std(method3)
        
        target_std = conditioning_std
        
        method1_ratio = abs(method1_std - target_std) / target_std if target_std > 0 else float('inf')
        method2_ratio = abs(method2_std - target_std) / target_std if target_std > 0 else float('inf')
        method3_ratio = abs(method3_std - target_std) / target_std if target_std > 0 else float('inf')
        
        if method1_ratio <= method2_ratio and method1_ratio <= method3_ratio:
            return method1, "method1_standard"
        elif method2_ratio <= method3_ratio:
            return method2, "method2_range_based"
        else:
            return method3, "method3_direct_mapping"

    def evaluate_full_dataset_pattern_focused(self, dataloader, dataset_name, epoch, labels=None):
        """
        PATTERN-FOCUSED: Evaluation with emphasis on correlation tracking
        """
        print(f"Evaluating full {dataset_name} dataset with PATTERN-FOCUSED approach...")
        
        all_concatenated_data = []
        all_mses = []
        all_correlations = []
        denorm_methods_used = []
        
        # Get original dataset
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
                
                # Generate predicted signal
                result = self.pipeline(
                    normalized_signal,
                    self.config.prediction_point,
                    batch_size=len(normalized_signal),
                    num_inference_steps=100,  # Moderate steps for balance
                )
                
                predicted_signal = result.images
                
                # Process each sample
                for i in range(len(normalized_signal)):
                    sample_idx = batch_idx * dataloader.batch_size + i
                    
                    if sample_idx >= total_samples:
                        continue
                    
                    # Get original data
                    original_data_full = original_dataset.data[sample_idx]
                    
                    # Get conditioning part
                    conditioning_part_original = original_data_full[:self.config.prediction_point]
                    
                    # Get predicted part
                    predicted_part_normalized = predicted_signal[i, 0, self.config.prediction_point:].cpu().numpy()
                    
                    # Fixed denormalization
                    predicted_part_denormalized, method_used = self.fixed_denormalize(
                        predicted_part_normalized, conditioning_part_original
                    )
                    denorm_methods_used.append(method_used)
                    
                    # Concatenate
                    full_signal_original_scale = np.concatenate([
                        conditioning_part_original,
                        predicted_part_denormalized
                    ])
                    
                    all_concatenated_data.append(full_signal_original_scale)
                    
                    # Calculate metrics
                    original_second_half = original_data_full[self.config.prediction_point:]
                    mse = np.mean((predicted_part_denormalized - original_second_half) ** 2)
                    all_mses.append(mse)
                    
                    # Calculate correlation
                    if len(original_second_half) > 1:
                        corr = np.corrcoef(original_second_half, predicted_part_denormalized)[0, 1]
                        all_correlations.append(corr if not np.isnan(corr) else 0)
                    else:
                        all_correlations.append(0)
                    
                    processed_samples += 1
                
                if batch_idx % 15 == 0:
                    print(f"Processed batch {batch_idx+1}/{len(dataloader)}, samples: {processed_samples}/{total_samples}")
        
        print(f"FINAL: Processed {processed_samples} samples out of {total_samples} total")
        
        # Convert to arrays
        all_concatenated_data = np.array(all_concatenated_data)
        all_mses = np.array(all_mses)
        all_correlations = np.array(all_correlations)
        
        # Print correlation statistics
        avg_correlation = np.mean(all_correlations)
        positive_corr_count = np.sum(all_correlations > 0)
        strong_corr_count = np.sum(all_correlations > 0.3)
        very_strong_corr_count = np.sum(all_correlations > 0.5)
        
        print(f"🎯 PATTERN-FOCUSED CORRELATION RESULTS:")
        print(f"   Average correlation: {avg_correlation:.6f}")
        print(f"   Positive correlations: {positive_corr_count} / {len(all_correlations)} ({100*positive_corr_count/len(all_correlations):.1f}%)")
        print(f"   Strong correlations (>0.3): {strong_corr_count} / {len(all_correlations)} ({100*strong_corr_count/len(all_correlations):.1f}%)")
        print(f"   Very strong correlations (>0.5): {very_strong_corr_count} / {len(all_correlations)} ({100*very_strong_corr_count/len(all_correlations):.1f}%)")
        
        # Classification potential assessment
        if avg_correlation > 0.3:
            classification_potential = "🟢 GOOD - Should work well for classification"
        elif avg_correlation > 0.1:
            classification_potential = "🟡 MODERATE - May work for classification with feature engineering"
        elif avg_correlation > 0:
            classification_potential = "🟠 POOR - Limited classification potential"
        else:
            classification_potential = "🔴 VERY POOR - Not suitable for classification"
        
        print(f"📊 CLASSIFICATION POTENTIAL: {classification_potential}")
        
        # Save dataset
        father_path = f"{self.config.output_dir}/{epoch}"
        if not os.path.exists(father_path):
            os.makedirs(father_path)
        
        csv_filename = f"{father_path}/concatenated_{dataset_name}_data_pattern_focused_epoch_{epoch}.csv"
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for sample in all_concatenated_data:
                writer.writerow(sample)
        
        print(f"✅ Saved pattern-focused {dataset_name} data to {csv_filename}")
        print(f"📊 Final shape: {all_concatenated_data.shape}")
        print(f"📊 Data range: {np.min(all_concatenated_data):.4f} to {np.max(all_concatenated_data):.4f}")
        
        # Log comprehensive statistics
        avg_mse = np.mean(all_mses)
        wandb.log({
            f'{dataset_name}_mse_pattern_focused': avg_mse,
            f'{dataset_name}_avg_correlation_pattern_focused': avg_correlation,
            f'{dataset_name}_positive_correlations_pattern_focused': positive_corr_count,
            f'{dataset_name}_strong_correlations_pattern_focused': strong_corr_count,
            f'{dataset_name}_very_strong_correlations_pattern_focused': very_strong_corr_count,
            f'{dataset_name}_classification_score': avg_correlation * 100,  # Simple score for tracking
        })
        
        return all_concatenated_data, all_mses

    def train(self):
        print("🚀 Starting PATTERN-FOCUSED training to fix negative correlation...")
        print("🎯 Goal: Achieve positive correlations >0.3 for good classification performance")
        
        # Print info
        print(f"📊 Training dataset: {len(self.train_dataset)} samples")
        print(f"📊 Validation dataset: {len(self.val_dataset)} samples")
        print(f"🔧 Pattern-focused: Lower noise (25%), correlation loss, pattern matching")
        
        number_iteration = 0
        best_correlation = -1.0  # Track best correlation instead of MSE
        eval_correlation_history = []

        for epoch in range(self.config.num_epochs):
            epoch_losses = []

            for iteration, batch in enumerate(self.train_dataloader):
                # Use pattern-focused training
                training_loss = self.train_single_batch_pattern_focused(batch, number_iteration)
                epoch_losses.append(training_loss)
                number_iteration += 1

                # Evaluation focused on correlation improvement
                eval_interval = 200

                if (number_iteration >= 400 and number_iteration % eval_interval == 0):

                    print(f"\n🔍 === PATTERN-FOCUSED Evaluation at iteration {number_iteration} ===")

                    # Pattern-focused evaluation
                    train_data, train_mses = self.evaluate_full_dataset_pattern_focused(
                        self.train_dataloader, "train", number_iteration, self.train_labels
                    )

                    val_data, val_mses = self.evaluate_full_dataset_pattern_focused(
                        self.val_dataloader, "test", number_iteration, self.test_labels
                    )

                    # Calculate correlation metrics for tracking
                    train_correlations = []
                    val_correlations = []
                    
                    for idx in range(min(len(train_data), len(self.train_dataset_original.data))):
                        orig = self.train_dataset_original.data[idx][self.config.prediction_point:]
                        pred = train_data[idx][self.config.prediction_point:]
                        if len(orig) > 1:
                            corr = np.corrcoef(orig, pred)[0, 1]
                            train_correlations.append(corr if not np.isnan(corr) else 0)
                    
                    for idx in range(min(len(val_data), len(self.val_dataset_original.data))):
                        orig = self.val_dataset_original.data[idx][self.config.prediction_point:]
                        pred = val_data[idx][self.config.prediction_point:]
                        if len(orig) > 1:
                            corr = np.corrcoef(orig, pred)[0, 1]
                            val_correlations.append(corr if not np.isnan(corr) else 0)
                    
                    train_avg_correlation = np.mean(train_correlations)
                    val_avg_correlation = np.mean(val_correlations)
                    
                    eval_correlation_history.append(val_avg_correlation)

                    # Enhanced logging
                    wandb.log({
                        "training_loss_pattern_focused": np.mean(epoch_losses[-10:]) if len(epoch_losses) >= 10 else np.mean(epoch_losses),
                        "train_correlation_pattern_focused": train_avg_correlation,
                        "val_correlation_pattern_focused": val_avg_correlation,
                        "correlation_improvement": val_avg_correlation - eval_correlation_history[0] if len(eval_correlation_history) > 1 else 0,
                        "iteration": number_iteration,
                    })

                    # Early stopping based on correlation improvement
                    if val_avg_correlation > best_correlation:
                        best_correlation = val_avg_correlation
                        torch.save(self.unet.state_dict(), 
                                 f"{self.config.output_dir}/best_model_pattern_focused.pth")
                        print(f"💾 New best correlation model: {best_correlation:.6f}")

                    # Progress summary
                    print(f"📊 PATTERN-FOCUSED Metrics:")
                    print(f"   Train Correlation: {train_avg_correlation:.6f}")
                    print(f"   Val Correlation: {val_avg_correlation:.6f}")
                    print(f"   Best Correlation: {best_correlation:.6f}")
                    print(f"   Positive Train: {np.sum(np.array(train_correlations) > 0)} / {len(train_correlations)}")
                    print(f"   Positive Val: {np.sum(np.array(val_correlations) > 0)} / {len(val_correlations)}")
                    print(f"=== PATTERN-FOCUSED Evaluation Complete ===\n")

            # Epoch summary
            epoch_avg_loss = np.mean(epoch_losses)
            print(f"📈 Epoch {epoch}: Avg Training Loss = {epoch_avg_loss:.6f}")

        print(f"🎉 PATTERN-FOCUSED training completed! Best correlation: {best_correlation:.6f}")

        # Final evaluation
        print("\n🏁 === Final PATTERN-FOCUSED Evaluation ===")
        final_train_data, final_train_mses = self.evaluate_full_dataset_pattern_focused(
           self.train_dataloader, "train_final", "final", self.train_labels
        )
        final_val_data, final_val_mses = self.evaluate_full_dataset_pattern_focused(
           self.val_dataloader, "test_final", "final", self.test_labels
        )

        print(f"📈 Final PATTERN-FOCUSED Results:")
        print(f"   Training: {final_train_data.shape}")
        print(f"   Validation: {final_val_data.shape}")
        print(f"   Best correlation achieved: {best_correlation:.6f}")
        print(f"   Best model: {self.config.output_dir}/best_model_pattern_focused.pth")

        # Final classification potential assessment
        if best_correlation > 0.3:
            print("🎯 CLASSIFICATION OUTLOOK: 🟢 GOOD - Expected AUC >0.7")
        elif best_correlation > 0.1:
            print("🎯 CLASSIFICATION OUTLOOK: 🟡 MODERATE - Expected AUC 0.6-0.7")
        elif best_correlation > 0:
            print("🎯 CLASSIFICATION OUTLOOK: 🟠 POOR - Expected AUC 0.5-0.6")
        else:
            print("🎯 CLASSIFICATION OUTLOOK: 🔴 VERY POOR - Expected AUC <0.5")

        wandb.log({
            "final_best_correlation_pattern_focused": best_correlation,
            "training_completed_pattern_focused": True
        })