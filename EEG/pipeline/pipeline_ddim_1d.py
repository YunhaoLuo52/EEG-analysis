from typing import List, Optional, Tuple, Union
import torch
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import DiffusionPipeline, ImagePipelineOutput

class DDIMPipeline1D(DiffusionPipeline):
    """
    DDIM Pipeline for 1D EEG data.
    
    Parameters:
        unet: U-Net architecture to denoise the encoded signal.
        scheduler: A scheduler to be used in combination with unet to denoise the encoded signal.
    """

    def __init__(self, unet, scheduler):
        super().__init__()

        # Convert scheduler to DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
            self,
            initial_signal,
            prediction_point,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            eta: float = 0.0,
            num_inference_steps: int = 1000,
            use_clipped_model_output: Optional[bool] = None,
            return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Args:
            initial_signal: The initial signal to condition on.
            prediction_point: Point from which to start prediction.
            batch_size: The number of samples to generate.
            generator: Torch generator(s) for deterministic generation.
            eta: Controls scale of variance (0 is DDIM, 1 is one type of DDPM).
            num_inference_steps: Number of denoising steps.
            use_clipped_model_output: Whether to clip model output.
            return_dict: Whether to return as a dictionary.

        Returns:
            Generated signal.
        """
        # Sample Gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            signal_shape = (batch_size, self.unet.config.in_channels, self.unet.config.sample_size)
        else:
            signal_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # # Update this part with scaled noise
        # noise_scale = 1  # Use same value as in training
        # signal = noise_scale * randn_tensor(signal_shape, generator=generator, 
        #                                 device=self.device, dtype=self.unet.dtype)
        # signal[:, :, :prediction_point] = initial_signal[:, :, :prediction_point]
        
        # FIXED: Generate noise that matches original signal characteristics
        original_mean = initial_signal[:, :, :prediction_point].mean()
        original_std = initial_signal[:, :, :prediction_point].std()

        noise = randn_tensor(signal_shape, generator=generator, device=self.device, dtype=self.unet.dtype)
        # Scale noise to match original signal statistics
        noise = noise * (original_std * 0.1) + original_mean  # 10% of original std

        signal = noise.clone()
        signal[:, :, :prediction_point] = initial_signal[:, :, :prediction_point]
        
        # Set step values
        self.scheduler.set_timesteps(num_inference_steps)


        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. Predict noise model_output
            model_output = self.unet(signal, t).sample

            # 2. Predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            signal = self.scheduler.step(
                model_output, t, signal, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample
            signal[:, :, :prediction_point] = initial_signal[:, :, :prediction_point]

            # ADD THIS CRITICAL BOUNDARY SMOOTHING CODE:
            transition_width = 10
            if prediction_point + transition_width < signal.shape[-1]:
                transition_weights = torch.linspace(0, 1, transition_width).to(signal.device)
                last_original_value = initial_signal[:, :, prediction_point-1:prediction_point]

                for i in range(transition_width):
                    weight = transition_weights[i]
                    signal[:, :, prediction_point + i] = (
                        (1 - weight) * last_original_value.squeeze(-1) +
                        weight * signal[:, :, prediction_point + i]
                    )
    

        
        # # Normalize signal
        # signal = (signal / 2 + 0.5).clamp(0, 1)
        
        # CRITICAL FIX: Don't force normalize to [0,1] - preserve original signal characteristics
        # Instead, clamp to reasonable range based on original signal
        signal_min = initial_signal.min()
        signal_max = initial_signal.max()

        # Only clamp the predicted part, keep original part unchanged
        predicted_part = signal[:, :, prediction_point:]
        predicted_part = torch.clamp(predicted_part, signal_min * 1.2, signal_max * 1.2)  # Allow 20% extension
        signal[:, :, prediction_point:] = predicted_part

        # Original part stays exactly the same
        signal[:, :, :prediction_point] = initial_signal[:, :, :prediction_point]


        if not return_dict:
            return (signal,)

        # Return as ImagePipelineOutput for compatibility
        return ImagePipelineOutput(images=signal)

    @torch.no_grad()
    def do_prediction(
            self,
            initial_signal,
            prediction_point: int,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            eta: float = 0.0,
            num_inference_steps: int = 100,
            use_clipped_model_output: Optional[bool] = None,
    ) -> torch.FloatTensor:
        """
        Performs prediction starting from a specific point.
        
        Args:
            initial_signal: The initial signal to condition on.
            predict_begin_point: Point from which to start prediction.
            batch_size: The number of samples to generate.
            generator: Torch generator(s) for deterministic generation.
            eta: Controls scale of variance (0 is DDIM, 1 is one type of DDPM).
            num_inference_steps: Number of denoising steps.
            use_clipped_model_output: Whether to clip model output.

        Returns:
            Predicted signal.
        """
        # # Clone the initial signal
        # signal = initial_signal.clone()

        # def do_prediction(self, initial_signal, predict_begin_point: int, ...):
        # Instead of just cloning, start with random noise like in __call__


        original_mean = initial_signal[:, :, :prediction_point].mean()
        original_std = initial_signal[:, :, :prediction_point].std()

        noise = randn_tensor(initial_signal.shape, generator=generator, 
                    device=self.device, dtype=self.unet.dtype)
        noise = noise * (original_std * 0.1) + original_mean

        signal = noise.clone()
        signal[:, :, :prediction_point] = initial_signal[:, :, :prediction_point]

        # Set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. Predict noise model_output
            model_output = self.unet(signal, t).sample

            # 2. Predict previous mean of signal x_t-1 and add variance depending on eta
            # signal = self.scheduler.step(
            #     model_output, t, signal, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            # ).prev_sample
            # signal[:, :, :prediction_point] = initial_signal[:, :, :prediction_point]
            signal = self.scheduler.step(
                model_output, t, signal, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

            # Always preserve conditioning
            signal[:, :, :prediction_point] = initial_signal[:, :, :prediction_point]

            # CRITICAL FIX: Add smooth transition at boundary
            transition_width = 10
            if prediction_point + transition_width < signal.shape[-1]:
                transition_weights = torch.linspace(0, 1, transition_width).to(signal.device)
                last_original_value = initial_signal[:, :, prediction_point-1:prediction_point]
    
                for i in range(transition_width):
                    weight = transition_weights[i]
                    signal[:, :, prediction_point + i] = (
                        (1 - weight) * last_original_value.squeeze(-1) +
                        weight * signal[:, :, prediction_point + i]
                    )

        # Preserve signal range like in __call__ method
        signal_min = initial_signal.min()
        signal_max = initial_signal.max()

        predicted_part = signal[:, :, prediction_point:]
        predicted_part = torch.clamp(predicted_part, signal_min * 1.2, signal_max * 1.2)
        signal[:, :, prediction_point:] = predicted_part

        # Ensure original part stays the same
        signal[:, :, :prediction_point] = initial_signal[:, :, :prediction_point]

        return signal