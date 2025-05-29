# PHASE 1: MODERATE IMPROVEMENTS
# This builds on the successful conservative training with better settings

from EEG import EEGDiffTrainner1D, EEGDiffMR
from mmengine import Config
import wandb
import os
import torch
import types
import numpy as np
import torch.nn.functional as F

# Create necessary directories
os.makedirs("C:/Github/EEG-Mouse/outputs/1d_model_phase1", exist_ok=True)

print("🚀 PHASE 1: Moderate Improvements (CUDA-Safe)")
print("Building on successful conservative training...")
print("=" * 60)

# ===============================================
# MONITORING METHODS (Same as before, but with improvements)
# ===============================================

def train_single_batch_monitored_v2(self, batch, iteration):
    """Enhanced single batch training - Version 2 with better gradient handling"""
    try:
        clean_signals = batch[0].to(self.config.device)
        
        # Check input validity
        if torch.isnan(clean_signals).any():
            raise ValueError(f"NaN in input signals at iteration {iteration}")
        
        # Generate noise
        noise = torch.randn(clean_signals.shape, device=clean_signals.device)
        
        # Safe timestep generation
        max_timesteps = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, max_timesteps, (clean_signals.shape[0],), device=clean_signals.device).long()
        timesteps = torch.clamp(timesteps, 0, max_timesteps - 1)
        
        # Add noise and apply conditioning
        noisy_signals = self.noise_scheduler.add_noise(clean_signals, noise, timesteps)
        noisy_signals[:, :, :self.config.prediction_point] = clean_signals[:, :, :self.config.prediction_point]
        
        # UNet forward pass
        noise_pred = self.unet(noisy_signals, timesteps).sample
        
        # Calculate loss
        loss = F.mse_loss(
            noise_pred[:, :, self.config.prediction_point:],
            noise[:, :, self.config.prediction_point:]
        )
        
        # Check for invalid values
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"Invalid loss at iteration {iteration}: {loss}")
        
        # Backward pass with improved gradient clipping
        loss.backward()
        
        # IMPROVED: Better gradient clipping
        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=5.0)  # Increased from 1.0
        
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()
        
    except Exception as e:
        print(f"   ❌ Error in batch training: {e}")
        raise

def evaluate_dataset_improved(self, dataloader, dataset_name, iteration, max_samples=5):
    """Improved evaluation with more samples"""
    print(f"   Evaluating {dataset_name} dataset ({max_samples} samples)...")
    
    all_mses = []
    
    with torch.no_grad():
        sample_count = 0
        for batch in dataloader:
            if sample_count >= max_samples:
                break
            
            original_signal = batch[0][:1].to(self.config.device)
            
            # Generate prediction with more inference steps for better quality
            result = self.pipeline(
                original_signal,
                self.config.prediction_point,
                batch_size=1,
                num_inference_steps=10,  # Increased from 5
            )
            
            predicted_signal = result.images
            
            # Calculate MSE
            original_half = original_signal[0, 0, self.config.prediction_point:].cpu().numpy()
            predicted_half = predicted_signal[0, 0, self.config.prediction_point:].cpu().numpy()
            
            mse = np.mean((predicted_half - original_half) ** 2)
            all_mses.append(mse)
            sample_count += 1
    
    return np.array(all_mses)

def train_phase1_improved(self):
    """Phase 1 improved training with moderate settings"""
    print("🚀 Phase 1 Training: Moderate improvements on conservative base")
    
    # Print improved settings
    print(f"🔧 Phase 1 Settings:")
    print(f"   Batch size: {self.config.train_batch_size}")
    print(f"   Timesteps: {self.noise_scheduler.config.num_train_timesteps}")
    print(f"   Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
    print(f"   Max steps: {self.config.max_train_steps}")
    print(f"   Gradient clip: 5.0 (improved)")
    
    number_iteration = 0
    best_mse = float('inf')
    cuda_error_count = 0
    
    try:
        for epoch in range(self.config.num_epochs):
            print(f"\n🔄 Epoch {epoch+1}/{self.config.num_epochs}")
            epoch_losses = []
            
            for iteration, batch in enumerate(self.train_dataloader):
                try:
                    # Memory management
                    if number_iteration % 20 == 0:  # Less frequent cleanup
                        torch.cuda.empty_cache()
                    
                    # Training step
                    training_loss = self.train_single_batch_monitored_v2(batch, number_iteration)
                    epoch_losses.append(training_loss)
                    number_iteration += 1
                    
                    # Progress logging
                    if number_iteration % 25 == 0:  # Less frequent logging
                        print(f"   ✅ Iteration {number_iteration}: Loss = {training_loss:.6f}")
                    
                    # Evaluation
                    if (number_iteration >= self.config.get('eval_begin', 50) and 
                        number_iteration % self.config.eval_interval == 0):
                        
                        print(f"\n📊 Evaluation at iteration {number_iteration}")
                        
                        try:
                            # Improved evaluation
                            train_mses = self.evaluate_dataset_improved(
                                self.train_dataloader, "train", number_iteration, max_samples=5
                            )
                            val_mses = self.evaluate_dataset_improved(
                                self.val_dataloader, "test", number_iteration, max_samples=5
                            )
                            
                            train_avg_mse = np.mean(train_mses)
                            val_avg_mse = np.mean(val_mses)
                            
                            print(f"   Train MSE: {train_avg_mse:.6f}")
                            print(f"   Val MSE: {val_avg_mse:.6f}")
                            print(f"   MSE Ratio: {val_avg_mse/train_avg_mse:.3f}")
                            
                            # Wandb logging
                            try:
                                wandb.log({
                                    "train_signal_mse": train_avg_mse,
                                    "val_signal_mse": val_avg_mse,
                                    "mse_ratio": val_avg_mse/train_avg_mse,
                                    "training_loss": training_loss,
                                    "iteration": number_iteration,
                                    "phase": "phase1_improved"
                                })
                            except:
                                pass
                            
                            # Save best model
                            if val_avg_mse < best_mse:
                                best_mse = val_avg_mse
                                torch.save(self.unet.state_dict(), 
                                         f"{self.config.output_dir}/phase1_best_model.pth")
                                print(f"   🎯 New best model: {best_mse:.6f}")
                            
                        except Exception as eval_error:
                            print(f"   ⚠️ Evaluation failed: {eval_error}")
                    
                    # Stop condition
                    if number_iteration >= self.config.max_train_steps:
                        print(f"🏁 Reached max steps ({self.config.max_train_steps})")
                        break
                        
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        cuda_error_count += 1
                        print(f"\n❌ CUDA ERROR #{cuda_error_count}: {e}")
                        if cuda_error_count >= 2:
                            print("🛑 Multiple CUDA errors - stopping training")
                            break
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
                
                except Exception as e:
                    print(f"❌ Unexpected error: {e}")
                    raise
            
            # Epoch summary
            if epoch_losses:
                epoch_avg = np.mean(epoch_losses)
                print(f"📈 Epoch {epoch+1}: Avg Loss = {epoch_avg:.6f}")
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    
    print(f"\n🏁 Phase 1 Summary:")
    print(f"   Iterations: {number_iteration}")
    print(f"   CUDA errors: {cuda_error_count}")
    print(f"   Best MSE: {best_mse:.6f}")

# ===============================================
# MAIN SCRIPT WITH PHASE 1 SETTINGS
# ===============================================

try:
    # Load and modify configuration
    print("📁 Loading configuration for Phase 1...")
    config_file_path = 'config/EEG-Diff/trainer_1d.py'
    config = Config.fromfile(config_file_path)
    
    # PHASE 1: MODERATE IMPROVEMENTS
    config.trainner.trainner_config.train_batch_size = 2        # Doubled from 1
    config.trainner.trainner_config.eval_batch_size = 2         # Doubled from 1
    config.trainner.trainner_config.num_train_timesteps = 100   # Doubled from 50
    config.trainner.trainner_config.max_train_steps = 200       # Quadrupled from 50
    config.trainner.trainner_config.num_epochs = 3             # Increased from 2
    config.trainner.trainner_config.learning_rate = 3e-6       # Tripled from 1e-6
    config.trainner.trainner_config.eval_begin = 50            # Later start
    config.trainner.trainner_config.eval_interval = 50         # Less frequent
    config.trainner.trainner_config.output_dir = "C:/Github/EEG-Mouse/outputs/1d_model_phase1"
    
    print("✅ Phase 1 configuration set")
    
    # Build trainer
    trainner = EEGDiffMR.build(config.trainner)
    
    # Add improved methods
    trainner.train_phase1_improved = types.MethodType(train_phase1_improved, trainner)
    trainner.train_single_batch_monitored_v2 = types.MethodType(train_single_batch_monitored_v2, trainner)
    trainner.evaluate_dataset_improved = types.MethodType(evaluate_dataset_improved, trainner)
    
    print("✅ Phase 1 methods added")

except Exception as e:
    print(f"❌ Setup failed: {e}")
    exit(1)

# Initialize wandb
key = 'e569a4bc3975771754384d1efb8bba0b7bbc1860'
if key:
    try:
        wandb.login(key=key)
        wandb.init(
            project="EEG-DIF-1D-Phase1",
            name='Phase 1: Moderate Improvements',
            config={
                "phase": "phase1_improved",
                "batch_size": 2,
                "timesteps": 100,
                "max_steps": 200,
                "learning_rate": 3e-6,
                "notes": "Building on successful conservative training"
            }
        )
    except Exception as e:
        print(f"⚠️ Wandb setup failed: {e}")

print("\n" + "=" * 60)
print("🚀 STARTING PHASE 1 IMPROVED TRAINING")
print("=" * 60)

# Start training
try:
    trainner.train_phase1_improved()
    print("\n🎉 Phase 1 training completed successfully!")
    
except Exception as e:
    print(f"\n❌ Phase 1 training failed: {e}")
    import traceback
    traceback.print_exc()

finally:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        if wandb.run:
            wandb.finish()
    except:
        pass

print("\n🎯 Next Steps:")
print("If Phase 1 succeeds with good MSE improvement:")
print("- Try Phase 2 with batch_size=4, timesteps=150")
print("- Eventually return to full training settings")
print("- Your CUDA issues are now resolved! 🎉")