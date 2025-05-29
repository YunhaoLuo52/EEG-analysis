from EEG import EEGDiffTrainner1D, EEGDiffMR
from mmengine import Config
import wandb
import os
import random
import numpy as np
import torch

def set_reproducible_seeds(seed=42):
    """Set all seeds for reproducible results"""
    print(f"🎲 Setting seeds to {seed} for reproducible results...")
    
    # Python random
    random.seed(seed)
    
    # Numpy random  
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print("✅ All seeds set for reproducible training")


# Add this line right after your imports, before building the trainer:
set_reproducible_seeds(seed=42)

# Create necessary directories
os.makedirs("C:/Github/EEG-Mouse/outputs/1d_model", exist_ok=True)
os.makedirs("C:/Github/EEG-Mouse/data", exist_ok=True)

# Load configuration
config_file_path = 'config/EEG-Diff/trainer_1d.py'
config = Config.fromfile(config_file_path)

# Build trainer
trainner = EEGDiffMR.build(config.trainner)

# Initialize wandb for tracking (optional)
# If you don't want to use wandb, comment out these lines
key = 'e569a4bc3975771754384d1efb8bba0b7bbc1860'  # Fill your wandb key here if you want to use it
if key:
    wandb.login(key=key)
    wandb.init(
        project=config.project_name,
        name='EEG-DIF-1D Training',
        config=config.wandb_config
    )
else:
    print("No wandb key provided, training without wandb logging")

# Start training
print("Starting training for 1D EEG diffusion model...")
trainner.train()
print("Training completed!")