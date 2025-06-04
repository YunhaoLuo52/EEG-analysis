from typing import List, Optional, Tuple, Union
import torch
import numpy as np
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import DiffusionPipeline, ImagePipelineOutput

class DDIMPipeline1D(DiffusionPipeline):
    """
    Pipeline for 1D EEG generation with z-score normalized data
    """

    def __init__(self, unet, scheduler):
        super().__init__()
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
        Generate EEG signals with z-score normalized data
        """
        if isinstance(self.unet.config.sample_size, int):
            signal_shape = (batch_size, self.unet.config.in_channels, self.unet.config.sample_size)
        else:
            signal_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        # Start with standard normal noise (appropriate for z-score data)
        signal = randn_tensor(signal_shape, generator=generator, device=self.device, dtype=self.unet.dtype)
        
        # Get conditioning part
        conditioning_part = initial_signal[:, :, :prediction_point]
        
        # Replace conditioning part
        signal[:, :, :prediction_point] = conditioning_part
        
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # Model prediction
            model_output = self.unet(signal, t).sample
            
            # Scheduler step
            signal = self.scheduler.step(
                model_output, t, signal, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample
            
            # Always preserve conditioning
            signal[:, :, :prediction_point] = conditioning_part
            
            # Boundary smoothing (only in final steps)
            if t < 50:  # Apply smoothing only near the end
                transition_width = min(15, (signal.shape[-1] - prediction_point) // 4)
                if prediction_point + transition_width < signal.shape[-1]:
                    # Simple moving average for smooth transition
                    kernel_size = 5
                    padding = kernel_size // 2
                    
                    # Extract boundary region
                    boundary_start = max(0, prediction_point - 10)
                    boundary_end = min(signal.shape[-1], prediction_point + transition_width)
                    boundary_region = signal[:, :, boundary_start:boundary_end].clone()
                    
                    # Apply smoothing
                    smooth_kernel = torch.ones(1, 1, kernel_size, device=signal.device, dtype=signal.dtype) / kernel_size
                    boundary_padded = torch.nn.functional.pad(boundary_region, (padding, padding), mode='reflect')
                    smoothed = torch.nn.functional.conv1d(boundary_padded, smooth_kernel, padding=0)
                    
                    # Blend only the generated part
                    blend_start = prediction_point - boundary_start
                    alpha = 0.3  # 30% smoothed, 70% original
                    signal[:, :, prediction_point:boundary_end] = (
                        alpha * smoothed[:, :, blend_start:] + 
                        (1 - alpha) * signal[:, :, prediction_point:boundary_end]
                    )

        # For z-score data, clip to reasonable range (e.g., ±4 standard deviations)
        signal = torch.clamp(signal, -4.0, 4.0)
        
        # Ensure conditioning unchanged
        signal[:, :, :prediction_point] = conditioning_part

        if not return_dict:
            return (signal,)

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
        Simplified prediction method for z-score normalized data
        """
        conditioning_part = initial_signal[:, :, :prediction_point]

        # Start with standard normal noise
        noise = randn_tensor(initial_signal.shape, generator=generator, 
                    device=self.device, dtype=self.unet.dtype)

        signal = noise.clone()
        signal[:, :, :prediction_point] = conditioning_part

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = self.unet(signal, t).sample

            signal = self.scheduler.step(
                model_output, t, signal, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

            signal[:, :, :prediction_point] = conditioning_part

        # Clip to reasonable range
        signal = torch.clamp(signal, -4.0, 4.0)
        signal[:, :, :prediction_point] = conditioning_part
        
        return signal