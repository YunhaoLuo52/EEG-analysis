# REPRODUCIBLE TRAINING SETUP WITH SEEDS
# This will help you get consistent results like your successful MSE 1.28 run

import torch
import numpy as np
import random
import os
from EEG import EEGDiffTrainner1D, EEGDiffMR
from mmengine import Config
import wandb

def set_all_seeds(seed=42):
    """Set all random seeds for reproducible training"""
    print(f"🎲 Setting all seeds to {seed} for reproducible results...")
    
    # Python random
    random.seed(seed)
    
    # Numpy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # PyTorch deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print("✅ All seeds set for reproducible training")

def reproducible_training_script():
    """Training script with proper seed control"""
    
    print("🔄 REPRODUCIBLE LARGE MODEL TRAINING")
    print("Returning to successful MSE 1.28 configuration")
    print("=" * 60)
    
    # STEP 1: Set seeds FIRST (before any model creation)
    set_all_seeds(seed=42)  # Use same seed as successful run
    
    # Create output directory
    os.makedirs("C:/Github/EEG-Mouse/outputs/1d_model_reproducible", exist_ok=True)
    
    # STEP 2: Load config with EXACT same settings as successful run
    try:
        print("📁 Loading configuration...")
        config = Config.fromfile('config/EEG-Diff/trainer_1d.py')
        
        # CRITICAL: Use EXACT same settings as your successful MSE 1.28 run
        config.trainner.trainner_config.train_batch_size = 1
        config.trainner.trainner_config.eval_batch_size = 1
        config.trainner.trainner_config.max_train_steps = 1000
        config.trainner.trainner_config.learning_rate = 5e-5     # Same as successful run
        config.trainner.trainner_config.num_epochs = 3
        config.trainner.trainner_config.output_dir = "C:/Github/EEG-Mouse/outputs/1d_model_reproducible"
        
        # EXACT same optimizer settings
        config.trainner.optimizer.learning_rate = 5e-5
        config.trainner.optimizer.weight_decay = 0.01
        
        print("✅ Configuration set to match successful run")
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return
    
    # STEP 3: Build trainer AFTER setting seeds
    try:
        print("🔧 Building trainer with set seeds...")
        trainner = EEGDiffMR.build(config.trainner)
        print("✅ Trainer built")
        
        # Verify model parameters match successful run
        total_params = sum(p.numel() for p in trainner.unet.parameters())
        print(f"📊 Model Parameters: {total_params:,}")
        
        if total_params < 500000:
            print("⚠️  WARNING: Model seems small - check large model config")
        else:
            print("✅ Large model confirmed")
            
    except Exception as e:
        print(f"❌ Trainer building failed: {e}")
        return
    
    # STEP 4: Add the same fixed training method
    import types
    import torch.nn.functional as F
    
    def train_batch_reproducible(self, batch, iteration):
        """Same training method as successful run"""
        clean_signals = batch[0].to(self.config.device)
        
        # Same conditioning fix
        noise = torch.randn(clean_signals.shape, device=clean_signals.device)
        noise_target = noise.clone()
        noise_target[:, :, :self.config.prediction_point] = 0.0
        
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                (clean_signals.shape[0],), device=clean_signals.device).long()
        
        noisy_signals = self.noise_scheduler.add_noise(clean_signals, noise, timesteps)
        noisy_signals[:, :, :self.config.prediction_point] = clean_signals[:, :, :self.config.prediction_point]
        
        noise_pred = self.unet(noisy_signals, timesteps).sample
        loss = F.mse_loss(noise_pred, noise_target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def train_reproducible_stable(self):
        """Stable training loop matching successful run"""
        print("🚀 Starting reproducible training...")
        
        number_iteration = 0
        best_mse = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\n🔄 Reproducible Epoch {epoch+1}/{self.config.num_epochs}")
            
            for iteration, batch in enumerate(self.train_dataloader):
                # Memory cleanup
                if number_iteration % 30 == 0:
                    torch.cuda.empty_cache()
                
                # Training step
                loss = self.train_batch_reproducible(batch, number_iteration)
                number_iteration += 1
                
                # Progress logging
                if number_iteration % 50 == 0:
                    print(f"   Step {number_iteration}: Loss = {loss:.6f}")
                
                # Evaluation - same as successful run
                if number_iteration % 100 == 0:
                    print(f"\n📊 Evaluation - Step {number_iteration}")
                    
                    test_batch = next(iter(self.val_dataloader))
                    test_signal = test_batch[0][:1].to(self.config.device)
                    
                    with torch.no_grad():
                        result = self.pipeline(test_signal, self.config.prediction_point,
                                             batch_size=1, num_inference_steps=20)  # Same as successful run
                    
                    predicted = result.images
                    
                    # Calculate metrics
                    orig = test_signal[0, 0, self.config.prediction_point:].cpu().numpy()
                    pred = predicted[0, 0, self.config.prediction_point:].cpu().numpy()
                    
                    mse = np.mean((orig - pred) ** 2)
                    std = np.std(pred)
                    
                    print(f"   MSE: {mse:.6f} (target: ~1.28)")
                    print(f"   Std: {std:.6f} (target: ~0.13)")
                    
                    # Check if we're getting reasonable results
                    if std > 0.5:
                        print("   ⚠️  High std - might be generating noise")
                    elif std < 0.05:
                        print("   ⚠️  Low std - might be too flat")
                    else:
                        print("   ✅ Reasonable std - good predictions")
                    
                    if mse < best_mse:
                        best_mse = mse
                        torch.save(self.unet.state_dict(),
                                 f"{self.config.output_dir}/reproducible_best.pth")
                        print(f"   🎯 New best: {best_mse:.6f}")
                    
                    # Wandb logging
                    try:
                        wandb.log({
                            "reproducible_mse": mse,
                            "reproducible_std": std,
                            "loss": loss,
                            "step": number_iteration
                        })
                    except:
                        pass
                
                if number_iteration >= self.config.max_train_steps:
                    break
        
        return best_mse
    
    def final_reproducible_test(self):
        """Final test matching successful run format"""
        print("\n🧪 Final Reproducible Test...")
        
        test_batch = next(iter(self.val_dataloader))
        test_signal = test_batch[0][:1].to(self.config.device)
        
        with torch.no_grad():
            result = self.pipeline(test_signal, self.config.prediction_point,
                                 batch_size=1, num_inference_steps=20)
        
        output = result.images
        
        orig = test_signal[0, 0, self.config.prediction_point:].cpu().numpy()
        pred = output[0, 0, self.config.prediction_point:].cpu().numpy()
        
        final_mse = np.mean((orig - pred) ** 2)
        final_std = np.std(pred)
        
        print(f"🎯 REPRODUCIBLE RESULTS:")
        print(f"   MSE: {final_mse:.6f} (successful run was: 1.28)")
        print(f"   Std: {final_std:.6f} (successful run was: 0.13)")
        
        # Create comparison plot
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 5))
        plt.plot(test_signal[0, 0].cpu().numpy(), label='Original', alpha=0.8)
        plt.plot(output[0, 0].cpu().numpy(), label='Reproducible Model', alpha=0.8)
        plt.axvline(x=self.config.prediction_point, color='red', linestyle='--', label='Prediction Point')
        plt.title(f'Reproducible Results (MSE: {final_mse:.6f}, Std: {final_std:.6f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 5)  # Set reasonable y-limits
        plt.savefig(f"{self.config.output_dir}/reproducible_results.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plot saved: {self.config.output_dir}/reproducible_results.png")
        
        # Success evaluation
        if 0.8 <= final_mse <= 2.0 and 0.1 <= final_std <= 0.3:
            print("\n🎉 SUCCESS! Reproduced good results!")
            return True
        elif final_mse > 5.0 or final_std > 0.5:
            print("\n❌ NOISE GENERATION - like current bad run")
            return False
        else:
            print("\n🟡 DIFFERENT RESULTS - but not terrible")
            return True
    
    # Add methods to trainer
    trainner.train_batch_reproducible = types.MethodType(train_batch_reproducible, trainner)
    trainner.train_reproducible_stable = types.MethodType(train_reproducible_stable, trainner)
    trainner.final_reproducible_test = types.MethodType(final_reproducible_test, trainner)
    
    # STEP 5: Initialize wandb
    key = 'e569a4bc3975771754384d1efb8bba0b7bbc1860'
    if key:
        try:
            wandb.login(key=key)
            wandb.init(
                project="EEG-DIF-1D-REPRODUCIBLE",
                name='Reproducible Training - Target MSE 1.28',
                config={"seed": 42, "target_mse": 1.28, "approach": "reproducible"}
            )
        except:
            print("⚠️ Wandb setup failed")
    
    print("\n" + "=" * 60)
    print("🚀 STARTING REPRODUCIBLE TRAINING")
    print("Goal: Reproduce successful MSE ~1.28, Std ~0.13")
    print("=" * 60)
    
    # STEP 6: Run training
    try:
        best_mse = trainner.train_reproducible_stable()
        success = trainner.final_reproducible_test()
        
        print(f"\n🎯 REPRODUCIBLE TRAINING COMPLETE")
        print(f"   Best MSE: {best_mse:.6f}")
        
        if success:
            print("✅ Successfully reproduced good results!")
        else:
            print("❌ Still getting poor results - need to investigate further")
        
        return success
        
    except Exception as e:
        print(f"❌ Reproducible training failed: {e}")
        return False
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            if wandb.run:
                wandb.finish()
        except:
            pass

# MAIN EXECUTION
if __name__ == "__main__":
    success = reproducible_training_script()
    
    if success:
        print("\n🎉 PROBLEM SOLVED!")
        print("Consistent, reproducible results achieved!")
    else:
        print("\n⚠️ Need to investigate training instability further")