from EEG import UNet1DModelWrapper

# Minimal, safe UNet1D configuration for single-channel EEG signals
unet = dict(
    type=UNet1DModelWrapper,
    
    # Core parameters (definitely supported)
    sample_size=1280,          # Keep your EEG signal length
    in_channels=1,             # Single channel EEG
    out_channels=1,            # Single channel output
    layers_per_block=2,        # Good balance for EEG complexity
    block_out_channels=(16, 32, 64, 128),  # More reasonable channel progression
    norm_num_groups=4,         # Better normalization than 4
    
    # Block types - keep your working configuration
    down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D", "DownBlock1D"),
    up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D", "UpBlock1D"),
    
    # Basic parameters that should be supported
    time_embedding_type="positional",
    act_fn="silu",
)