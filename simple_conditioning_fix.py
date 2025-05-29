# SIMPLE CONDITIONING FIX SCRIPT
# This retrains your UNet to properly respect conditioning

from EEG import EEGDiffTrainner1D, EEGDiffMR
from mmengine import Config
import torch
import torch.nn.functional as F
import types
import numpy as np

print("🔧 CONDITIONING FIX - Retraining UNet properly")
print("=" * 60)

def train_single_batch_conditioning_fix(self, batch, iteration):
    """Fixed training that teaches UNet to respect conditioning"""
    clean_signals = batch[0].to(self.config.device)
    
    # Generate noise for entire signal
    noise = torch.randn(clean_signals.shape, device=clean_signals.device)
    
    # CRITICAL FIX: Create proper noise target
    # Conditioning region should have ZERO noise target
    noise_target = noise.clone()
    noise_target[:, :, :self.config.prediction_point] = 0.0  # Zero for conditioning
    
    # Create timesteps
    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                            (clean_signals.shape[0],), device=clean_signals.device).long()
    
    # Add noise and apply conditioning
    noisy_signals = self.noise_scheduler.add_noise(clean_signals, noise, timesteps)
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
    
    return loss.item()

def conditioning_fix_training(self):
    """Quick conditioning fix training"""
    print("🚀 Starting conditioning fix training...")
    
    # Reinitialize UNet for clean training
    print("🔄 Reinitializing UNet...")
    for module in self.unet.modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if len(module.weight.shape) > 1:
                torch.nn.init.xavier_uniform_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    
    number_iteration = 0
    best_conditioning_error = float('inf')
    
    for epoch in range(2):  # Just 2 epochs for fix
        print(f"\n🔄 Fix Epoch {epoch+1}/2")
        
        for iteration, batch in enumerate(self.train_dataloader):
            # Use fixed training method
            loss = self.train_single_batch_conditioning_fix(batch, number_iteration)
            number_iteration += 1
            
            if number_iteration % 20 == 0:
                print(f"   Step {number_iteration}: Loss = {loss:.6f}")
            
            # Test conditioning every 50 steps
            if number_iteration % 50 == 0:
                # Quick conditioning test
                test_batch = next(iter(self.val_dataloader))
                test_signal = test_batch[0][:1].to(self.config.device)
                
                noise_input = torch.randn_like(test_signal)
                conditioned_input = noise_input.clone()
                conditioned_input[:, :, :self.config.prediction_point] = test_signal[:, :, :self.config.prediction_point]
                
                timestep = torch.tensor([50], device=self.config.device)
                with torch.no_grad():
                    noise_pred = self.unet(conditioned_input, timestep).sample
                
                conditioning_error = torch.mean(noise_pred[:, :, :self.config.prediction_point] ** 2)
                print(f"   Conditioning error: {conditioning_error:.6f}")
                
                if conditioning_error < best_conditioning_error:
                    best_conditioning_error = conditioning_error
                    torch.save(self.unet.state_dict(), f"{self.config.output_dir}/conditioning_fixed.pth")
                
                if conditioning_error < 0.01:
                    print("   ✅ Conditioning fixed!")
                    return True
            
            if number_iteration >= 300:  # Quick fix training
                break
    
    print(f"\n🏁 Conditioning fix complete. Best error: {best_conditioning_error:.6f}")
    return best_conditioning_error < 0.1

def test_fix_results(self):
    """Test the results after conditioning fix"""
    print("\n🧪 Testing fix results...")
    
    # Test with real signal
    test_batch = next(iter(self.val_dataloader))
    test_signal = test_batch[0][:1].to(self.config.device)
    
    # Generate prediction with fixed model
    result = self.pipeline(test_signal, self.config.prediction_point, batch_size=1, num_inference_steps=20)
    output = result.images
    
    # Check prediction quality
    original_pred_region = test_signal[0, 0, self.config.prediction_point:].cpu().numpy()
    generated_pred_region = output[0, 0, self.config.prediction_point:].cpu().numpy()
    
    mse = np.mean((original_pred_region - generated_pred_region) ** 2)
    
    print(f"Signal MSE after fix: {mse:.6f}")
    print(f"Prediction std: {np.std(generated_pred_region):.6f}")
    print(f"Prediction range: [{generated_pred_region.min():.4f}, {generated_pred_region.max():.4f}]")
    
    # Visual test
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    plt.plot(test_signal[0, 0].cpu().numpy(), label='Original', alpha=0.7)
    plt.plot(output[0, 0].cpu().numpy(), label='After Conditioning Fix', alpha=0.7)
    plt.axvline(x=self.config.prediction_point, color='red', linestyle='--', label='Prediction Point')
    plt.title(f'Results After Conditioning Fix (MSE: {mse:.6f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{self.config.output_dir}/after_conditioning_fix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Results plot saved to: {self.config.output_dir}/after_conditioning_fix.png")
    
    return mse

# MAIN EXECUTION
try:
    # Load trainer
    config = Config.fromfile('config/EEG-Diff/trainer_1d.py')
    trainner = EEGDiffMR.build(config.trainner)
    
    # Add methods
    trainner.train_single_batch_conditioning_fix = types.MethodType(train_single_batch_conditioning_fix, trainner)
    trainner.conditioning_fix_training = types.MethodType(conditioning_fix_training, trainner)
    trainner.test_fix_results = types.MethodType(test_fix_results, trainner)
    
    print("✅ Methods added")
    
    # Run conditioning fix
    fix_success = trainner.conditioning_fix_training()
    
    # Test results
    final_mse = trainner.test_fix_results()
    
    if fix_success and final_mse < 0.5:
        print("\n🎉 CONDITIONING FIX SUCCESS!")
        print(f"Final MSE: {final_mse:.6f} (should be much better than 1.22)")
        print("\nNow you can:")
        print("1. Use the fixed model for normal training")
        print("2. Expect much better signal completion")
        print("3. See natural EEG patterns instead of flat predictions")
    else:
        print(f"\n⚠️ Partial success. MSE: {final_mse:.6f}")
        print("May need longer conditioning fix training")

except Exception as e:
    print(f"❌ Conditioning fix failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🎯 After this fix, retrain normally and you should see:")
print("- Signal MSE: 1.22 → 0.1-0.3 (dramatic improvement)")
print("- Natural EEG predictions instead of flat noise")
print("- Smooth continuity at prediction boundary")
print("=" * 60)