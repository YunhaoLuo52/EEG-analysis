from mmengine import read_base
from EEG import EEGDiffTrainner1D

with read_base():
    from ._base_.train_dataset_1d import train_dataset, val_dataset
    from ._base_.unet_1d import unet
    from ._base_.noise_scheduler import noise_scheduler
    from ._base_.basic_information import prject_name

device = "cuda"  # Change to "cpu" if no GPU available

train_config = dict(
    device=device,
    output_dir="C:/Github/EEG-Mouse/outputs/1d_model",
    u_net_weight_path=None,  # Set to path if you have pretrained weights

    # EEG-specific parameters
    prediction_point=640,   # Start prediction from 50% of signal
    num_train_timesteps=1000, #number of steps in evaluation... ################### Denoise
    num_epochs=10,
    max_train_steps=None,
    train_batch_size=32,     # Adjust based on your GPU memory
    eval_batch_size=32,

    # Learning rate and scheduling
    learning_rate=5e-5,
    lr_warmup_steps=500,
    lr_scheduler="cosine",

    # Evaluation and saving
    eval_begin=200,          # Start evaluation after this many iterations
    eval_interval=200,       # Evaluate every 20 iterations
    save_interval=1000,
    
    # Add paths to label files for seizure/non-seizure classification
    train_labels_path="C:/Github/EEG-Mouse/data/train_labels.csv",
    test_labels_path="C:/Github/EEG-Mouse/data/test_labels.csv",
)

# OPTIMIZER OPTIMIZED FOR LOW LOSS
optimizer_config = dict(
    type="AdamW",
    learning_rate=5e-5,       # Higher LR to escape loss plateau
    weight_decay=0.01,       # Reduced for better fitting
    betas=(0.9, 0.999),        # Higher beta2 for smoother updates
    eps=1e-8,                 # Smaller eps for more precise updates
)
project_name = "EEG-DIF-1D"

trainner = dict(
    type=EEGDiffTrainner1D,
    trainner_config=train_config,
    unet=unet,
    noise_scheduler=noise_scheduler,
    optimizer=optimizer_config,  # Same as in train_config
    train_dataset=train_dataset,
    val_dataset=val_dataset)

# WANDB LOGGING CONFIGURATION
wandb_config = dict(
    learning_rate=optimizer_config['learning_rate'],
    architecture="1D EEG Diffusion Model",
    dataset="EEG-DIF-1D",
    epochs=train_config['num_epochs'],
    batch_size=train_config['train_batch_size'],
    noise_schedule="cosine",
    prediction_point=train_config['prediction_point'],
    notes="Optimized parameters for EEG signal generation"
)

