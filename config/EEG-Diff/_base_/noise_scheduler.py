from EEG import DDIMScheduler
noise_scheduler=dict(
        type=DDIMScheduler,
        
        # Timesteps - MUST match training config
        num_train_timesteps=1000,

        # Beta schedule - optimized for EEG signals
        beta_schedule="scaled_linear",
        beta_start=0.0001,
        beta_end=0.02,


        # Prediction type - MUST match training config
        prediction_type="epsilon",       # Match with trainer config (was "v_prediction")
    
        # Clipping and scaling
        clip_sample=False,              # Don't clip for EEG signals (was True)
    
        # Alpha scheduling
        set_alpha_to_one=False,         # Better convergence (was True)
        steps_offset=0,                 # Standard offset (was 1)
    
        # Training parameters
        trained_betas=None,             # Use computed betas

    )

