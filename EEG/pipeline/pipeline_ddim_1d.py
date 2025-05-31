from typing import List, Optional, Tuple, Union
import torch
import numpy as np
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import DiffusionPipeline, ImagePipelineOutput

class DDIMPipeline1D(DiffusionPipeline):
    """
    BALANCED: Pipeline for 7Hz EEG with moderate noise levels - not too high, not too low.
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
        BALANCED: Moderate noise levels for 7Hz EEG - enough to learn patterns, not too much to destroy smoothness.
        """
        if isinstance(self.unet.config.sample_size, int):
            signal_shape = (batch_size, self.unet.config.in_channels, self.unet.config.sample_size)
        else:
            signal_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # BALANCED: Moderate noise for 7Hz EEG (between too high and too low)
        conditioning_part = initial_signal[:, :, :prediction_point]
        
        conditioning_mean = conditioning_part.mean(dim=-1, keepdim=True)
        conditioning_std = conditioning_part.std(dim=-1, keepdim=True)
        
        # BALANCED: 3% noise level - enough variation to learn, low enough for 7Hz smoothness
        noise = randn_tensor(signal_shape, generator=generator, device=self.device, dtype=self.unet.dtype)
        noise = noise * (conditioning_std * 0.07) + conditioning_mean  # 3% instead of 1% or 10%
        
        signal = noise.clone()
        signal[:, :, :prediction_point] = conditioning_part
        
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = self.unet(signal, t).sample

            signal = self.scheduler.step(
                model_output, t, signal, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample
            
            # Always preserve conditioning
            signal[:, :, :prediction_point] = conditioning_part

            # MODERATE transition smoothing for 7Hz
            transition_width = min(30, (signal.shape[-1] - prediction_point) // 3)  # Moderate transition
            if prediction_point + transition_width < signal.shape[-1]:
                transition_indices = torch.arange(transition_width, device=signal.device, dtype=torch.float32)
                # Simple cosine transition - not overly complex
                transition_weights = 0.5 * (1 - torch.cos(torch.pi * transition_indices / (transition_width - 1)))
                
                # Get trend from last few conditioning values
                trend_window = min(20, conditioning_part.shape[-1])
                recent_values = conditioning_part[:, :, -trend_window:]
                trend_mean = recent_values.mean(dim=-1, keepdim=True)
                
                for i in range(transition_width):
                    weight = transition_weights[i].item()
                    signal[:, :, prediction_point + i] = (
                        (1 - weight) * trend_mean.squeeze(-1) +
                        weight * signal[:, :, prediction_point + i]
                    )

        # MODERATE clamping for 7Hz EEG
        predicted_part = signal[:, :, prediction_point:]
        
        # Use reasonable bounds based on conditioning statistics
        conditioning_mean_val = conditioning_part.mean()
        conditioning_std_val = conditioning_part.std()
        
        # Allow some variation but keep reasonable for 7Hz
        lower_bound = conditioning_mean_val - 3.0 * conditioning_std_val
        upper_bound = conditioning_mean_val + 3.0 * conditioning_std_val
        
        predicted_part = torch.clamp(predicted_part, lower_bound, upper_bound)
        signal[:, :, prediction_point:] = predicted_part

        # LIGHT smoothing (not excessive) for 7Hz characteristics
        if signal.shape[-1] - prediction_point > 4:
            pred_region = signal[:, :, prediction_point:].clone()
            kernel_size = 3  # Smaller kernel to preserve some variation
            padding = kernel_size // 2
            
            smooth_kernel = torch.ones(1, 1, kernel_size, device=signal.device, dtype=signal.dtype) / kernel_size
            pred_region_padded = torch.nn.functional.pad(pred_region, (padding, padding), mode='reflect')
            smoothed = torch.nn.functional.conv1d(pred_region_padded, smooth_kernel, padding=0)
            
            # Blend smoothed with original to preserve some variation
            alpha = 0.3  # 70% smoothed, 30% original
            signal[:, :, prediction_point:] = alpha * smoothed + (1 - alpha) * pred_region

        # Ensure conditioning part unchanged
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
        BALANCED: Moderate noise prediction for 7Hz EEG.
        """
        conditioning_part = initial_signal[:, :, :prediction_point]
        conditioning_mean = conditioning_part.mean(dim=-1, keepdim=True)
        conditioning_std = conditioning_part.std(dim=-1, keepdim=True)

        noise = randn_tensor(initial_signal.shape, generator=generator, 
                    device=self.device, dtype=self.unet.dtype)
        # BALANCED: 3% noise level
        noise = noise * (conditioning_std * 0.07) + conditioning_mean

        signal = noise.clone()
        signal[:, :, :prediction_point] = conditioning_part

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = self.unet(signal, t).sample

            signal = self.scheduler.step(
                model_output, t, signal, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

            signal[:, :, :prediction_point] = conditioning_part

            # Moderate transition
            transition_width = min(25, (signal.shape[-1] - prediction_point) // 4)
            if prediction_point + transition_width < signal.shape[-1]:
                transition_indices = torch.arange(transition_width, device=signal.device, dtype=torch.float32)
                transition_weights = 0.5 * (1 - torch.cos(torch.pi * transition_indices / (transition_width - 1)))
                
                trend_window = min(15, conditioning_part.shape[-1])
                recent_values = conditioning_part[:, :, -trend_window:]
                trend_mean = recent_values.mean(dim=-1, keepdim=True)
                
                for i in range(transition_width):
                    weight = transition_weights[i].item()
                    signal[:, :, prediction_point + i] = (
                        (1 - weight) * trend_mean.squeeze(-1) +
                        weight * signal[:, :, prediction_point + i]
                    )

        # Moderate clamping
        predicted_part = signal[:, :, prediction_point:]
        conditioning_mean_val = conditioning_part.mean()
        conditioning_std_val = conditioning_part.std()
        
        lower_bound = conditioning_mean_val - 3.0 * conditioning_std_val
        upper_bound = conditioning_mean_val + 3.0 * conditioning_std_val
        
        predicted_part = torch.clamp(predicted_part, lower_bound, upper_bound)
        signal[:, :, prediction_point:] = predicted_part

        # Light smoothing
        if signal.shape[-1] - prediction_point > 4:
            pred_region = signal[:, :, prediction_point:].clone()
            kernel_size = 3
            padding = kernel_size // 2
            
            smooth_kernel = torch.ones(1, 1, kernel_size, device=signal.device, dtype=signal.dtype) / kernel_size
            pred_region_padded = torch.nn.functional.pad(pred_region, (padding, padding), mode='reflect')
            smoothed = torch.nn.functional.conv1d(pred_region_padded, smooth_kernel, padding=0)
            
            alpha = 0.3
            signal[:, :, prediction_point:] = alpha * smoothed + (1 - alpha) * pred_region

        signal[:, :, :prediction_point] = conditioning_part
        return signal