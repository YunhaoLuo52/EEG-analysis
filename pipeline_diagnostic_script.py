# PRIORITY: Pipeline Diagnostic Script
# Run this BEFORE any more training to identify pipeline bugs

from EEG import EEGDiffTrainner1D, EEGDiffMR
from mmengine import Config
import torch
import numpy as np
import matplotlib.pyplot as plt
import types

print("🔍 PIPELINE DIAGNOSTIC - Finding the Root Cause")
print("=" * 60)

def debug_pipeline_comprehensively(self):
    """Comprehensive pipeline debugging to find root cause"""
    print("🔍 COMPREHENSIVE PIPELINE DIAGNOSTIC")
    print("=" * 50)
    
    # Test 1: Simple linear signal test
    print("TEST 1: Linear Signal Test")
    test_signal = torch.linspace(-1, 1, 1280).unsqueeze(0).unsqueeze(0).to(self.config.device)
    print(f"  Input: Linear signal from {test_signal.min():.2f} to {test_signal.max():.2f}")
    
    result = self.pipeline(test_signal, self.config.prediction_point, batch_size=1, num_inference_steps=10)
    output = result.images
    
    # Check conditioning preservation
    input_conditioning = test_signal[0, 0, :self.config.prediction_point]
    output_conditioning = output[0, 0, :self.config.prediction_point]
    conditioning_error = torch.mean((input_conditioning - output_conditioning) ** 2)
    
    print(f"  Conditioning Error: {conditioning_error:.8f}")
    if conditioning_error > 1e-6:
        print("  🚨 CRITICAL: Pipeline corrupts conditioning region!")
        print(f"     Input first 5: {input_conditioning[:5].cpu().numpy()}")
        print(f"     Output first 5: {output_conditioning[:5].cpu().numpy()}")
        return False
    else:
        print("  ✅ Conditioning preserved correctly")
    
    # Check prediction region
    output_prediction = output[0, 0, self.config.prediction_point:]
    pred_std = output_prediction.std()
    pred_range = output_prediction.max() - output_prediction.min()
    
    print(f"  Prediction std: {pred_std:.6f}")
    print(f"  Prediction range: {pred_range:.6f}")
    
    if pred_std < 0.01:
        print("  🚨 CRITICAL: Predictions are too uniform (low variance)!")
        return False
    
    # Test 2: Real EEG signal test
    print("\nTEST 2: Real EEG Signal Test")
    real_batch = next(iter(self.val_dataloader))
    real_signal = real_batch[0][:1].to(self.config.device)
    
    print(f"  Real signal range: [{real_signal.min():.4f}, {real_signal.max():.4f}]")
    
    result_real = self.pipeline(real_signal, self.config.prediction_point, batch_size=1, num_inference_steps=10)
    output_real = result_real.images
    
    # Check conditioning for real signal
    real_input_cond = real_signal[0, 0, :self.config.prediction_point]
    real_output_cond = output_real[0, 0, :self.config.prediction_point]
    real_cond_error = torch.mean((real_input_cond - real_output_cond) ** 2)
    
    print(f"  Real signal conditioning error: {real_cond_error:.8f}")
    
    # Check prediction quality
    real_output_pred = output_real[0, 0, self.config.prediction_point:]
    real_pred_std = real_output_pred.std()
    
    print(f"  Real prediction std: {real_pred_std:.6f}")
    print(f"  Real prediction range: [{real_output_pred.min():.4f}, {real_output_pred.max():.4f}]")
    
    # Test 3: Manual DDIM step test
    print("\nTEST 3: Manual DDIM Process Test")
    
    # Start with noise
    noise_signal = torch.randn_like(real_signal)
    # Apply conditioning
    conditioned_noise = noise_signal.clone()
    conditioned_noise[:, :, :self.config.prediction_point] = real_signal[:, :, :self.config.prediction_point]
    
    # Single denoising step
    timestep = torch.tensor([100], device=self.config.device)
    
    with torch.no_grad():
        noise_pred = self.unet(conditioned_noise, timestep).sample
    
    print(f"  UNet input range: [{conditioned_noise.min():.4f}, {conditioned_noise.max():.4f}]")
    print(f"  UNet output range: [{noise_pred.min():.4f}, {noise_pred.max():.4f}]")
    print(f"  UNet output std: {noise_pred.std():.6f}")
    
    # Check if UNet respects conditioning
    unet_cond_region = noise_pred[0, 0, :self.config.prediction_point]
    original_cond_region = real_signal[0, 0, :self.config.prediction_point]
    unet_cond_error = torch.mean((unet_cond_region - original_cond_region) ** 2)
    
    print(f"  UNet conditioning respect: {unet_cond_error:.8f}")
    
    if unet_cond_error > 0.1:
        print("  🚨 CRITICAL: UNet doesn't respect conditioning!")
        print("  This means the UNet is trained to modify the conditioning region")
        print("  Solution: Retrain with proper conditioning loss")
        return False
    
    # Create diagnostic plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Linear test
    plt.subplot(3, 1, 1)
    plt.plot(test_signal[0, 0].cpu().numpy(), label='Input Linear Signal', alpha=0.7)
    plt.plot(output[0, 0].cpu().numpy(), label='Pipeline Output', alpha=0.7)
    plt.axvline(x=self.config.prediction_point, color='red', linestyle='--', label='Prediction Point')
    plt.title('Test 1: Linear Signal Pipeline Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Real EEG test
    plt.subplot(3, 1, 2)
    plt.plot(real_signal[0, 0].cpu().numpy(), label='Real EEG Signal', alpha=0.7)
    plt.plot(output_real[0, 0].cpu().numpy(), label='Pipeline Output', alpha=0.7)
    plt.axvline(x=self.config.prediction_point, color='red', linestyle='--', label='Prediction Point')
    plt.title('Test 2: Real EEG Signal Pipeline Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: UNet direct test
    plt.subplot(3, 1, 3)
    plt.plot(conditioned_noise[0, 0].cpu().numpy(), label='UNet Input (Conditioned Noise)', alpha=0.7)
    plt.plot(noise_pred[0, 0].cpu().numpy(), label='UNet Output (Noise Prediction)', alpha=0.7)
    plt.axvline(x=self.config.prediction_point, color='red', linestyle='--', label='Prediction Point')
    plt.title('Test 3: UNet Direct Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{self.config.output_dir}/pipeline_diagnostic.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Diagnostic plot saved to: {self.config.output_dir}/pipeline_diagnostic.png")
    
    # Summary
    print(f"\n📋 DIAGNOSTIC SUMMARY:")
    print(f"   Linear test conditioning error: {conditioning_error:.8f}")
    print(f"   Real EEG conditioning error: {real_cond_error:.8f}")
    print(f"   UNet conditioning respect: {unet_cond_error:.8f}")
    print(f"   Prediction variance: {real_pred_std:.6f}")
    
    if conditioning_error > 1e-6 or real_cond_error > 1e-6:
        print("\n🚨 PIPELINE ISSUE DETECTED:")
        print("   - Conditioning region not preserved")
        print("   - Pipeline implementation has bugs")
        print("   - Training improvements will be limited until this is fixed")
        return False
    
    if real_pred_std < 0.01:
        print("\n🚨 PREDICTION QUALITY ISSUE:")
        print("   - Predictions are too uniform")
        print("   - Model architecture too small or undertrained")
        print("   - Need larger model or more training")
        return False
    
    print("\n✅ Pipeline appears to work correctly")
    print("   Issue might be model size or training duration")
    return True

# Load your existing trainer
try:
    config = Config.fromfile('config/EEG-Diff/trainer_1d.py')
    trainner = EEGDiffMR.build(config.trainner)
    
    # Add diagnostic method
    trainner.debug_pipeline_comprehensively = types.MethodType(debug_pipeline_comprehensively, trainner)
    
    print("✅ Trainer loaded successfully")
    
    # Run comprehensive diagnostic
    pipeline_ok = trainner.debug_pipeline_comprehensively()
    
    if pipeline_ok:
        print("\n🎯 CONCLUSION: Pipeline works - try larger model/more training")
        print("\nNext steps:")
        print("1. Use larger UNet: block_out_channels=(64, 128, 256)")
        print("2. Higher learning rate: 1e-4")  
        print("3. More training steps: 1000+")
        print("4. Longer sequences or different architecture")
    else:
        print("\n🚨 CONCLUSION: Pipeline has fundamental bugs")
        print("\nRequired fixes:")
        print("1. Fix DDIMPipeline1D implementation")
        print("2. Ensure conditioning region preservation")
        print("3. Check UNet training methodology")
        print("4. Verify noise scheduler compatibility")
        
except Exception as e:
    print(f"❌ Diagnostic failed: {e}")
    import traceback
    traceback.print_exc()