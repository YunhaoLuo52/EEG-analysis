from EEG import DDIMScheduler
noise_scheduler=dict(
        type=DDIMScheduler,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        beta_start=0.0001,
        clip_sample=True,
        clip_sample_range=1.0,
        num_train_timesteps=200,
        prediction_type="epsilon",
        set_alpha_to_one=False,
        steps_offset=0,
        trained_betas=None,
    )