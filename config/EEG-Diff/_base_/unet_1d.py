from EEG import UNet1DModelWrapper

unet = dict(
    type=UNet1DModelWrapper,
    sample_size=1280,  # Length of your EEG signals
    in_channels=1,     # Single channel EEG data
    out_channels=1,    # Output also single channel
    layers_per_block=1,
    block_out_channels=(16, 32),
    norm_num_groups=8,  # Reduced for 1D signals
    down_block_types=("DownBlock1D", "DownBlock1D"),
    up_block_types=("UpBlock1D", "UpBlock1D"),

    #mid_block_type="MidBlock1D",
    # Additional parameters supported by diffusers.UNet1DModel
    time_embedding_type="positional",
    act_fn="silu",
    #attention_head_dim=None,  # No attention for 1D EEG signals
    #norm_eps=1e-5,
    #resnet_time_scale_shift="default",
)