from typing import List, Optional, Tuple, Union
import torch
import numpy as np
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import DiffusionPipeline, ImagePipelineOutput
from scipy.signal import butter, sosfilt

def band_limited_noise(shape, fs=256, fmin=6, fmax=8, device='cpu', dtype=torch.float32):
    """
    Generate band-limited noise in a specific frequency band (e.g., around 7 Hz).
    shape: (batch, channels, length)
    fs: sampling frequency
    fmin/fmax: min and max frequency of noise band (Hz)
    """
    batch, channels, length = shape
    noise = np.random.randn(batch * channels, length)
    
    # Butterworth bandpass filter
    sos = butter(N=4, Wn=[fmin, fmax], btype='bandpass', fs=fs, output='sos')
    filtered = np.array([sosfilt(sos, x) for x in noise])
    
    filtered = filtered.reshape(batch, channels, length)
    return torch.tensor(filtered, dtype=dtype, device=device)

def pink_noise(shape):
    """
    Generate pink noise using 1/f scaling in frequency domain.
    """
    batch, channels, time = shape
    uneven = time % 2
    X = np.random.randn(batch * channels, time + uneven)
    X_fft = np.fft.rfft(X)
    S = np.arange(1, X_fft.shape[1] + 1)
    S = np.sqrt(1.0 / S)
    X_fft *= S
    pink = np.fft.irfft(X_fft)
    return torch.tensor(pink[:, :time].reshape(batch, channels, time), dtype=torch.float32)


def generate_eeg_like_noise(shape, label_tensor, fs=256, device='cpu', dtype=torch.float32):
    """
    Generate EEG-like noise with band-limited + pink noise. Slightly stronger if seizure.
    label_tensor: shape (batch,) with values 0 (non-seizure) or 1 (seizure)
    """
    batch, channels, length = shape
    base_bl = band_limited_noise(shape, fs=fs, fmin=4, fmax=12, device=device, dtype=dtype)
    base_pink = pink_noise(shape).to(device=device, dtype=dtype)

    label_tensor = torch.as_tensor(label_tensor, dtype=torch.float32, device=device).reshape(-1, 1, 1)

    # More noise for seizure (helps model learn to denoise seizure signals carefully)
    noise = base_bl + 0.05 * base_pink + 0.1 * label_tensor * torch.randn_like(base_bl)
    return noise


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
            initial_labels=None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Generate EEG signals with z-score normalized data
        """
        if isinstance(self.unet.config.sample_size, int):
            signal_shape = (batch_size, self.unet.config.in_channels, self.unet.config.sample_size)
        else:
            signal_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        # Custom noise
        signal = generate_eeg_like_noise(signal_shape, label_tensor=initial_labels, fs=256, device=self.device, dtype=self.unet.dtype)
        
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
            initial_labels=None,
    ) -> torch.FloatTensor:
        """
        Simplified prediction method for z-score normalized data
        """
        conditioning_part = initial_signal[:, :, :prediction_point]

        # Custom noise
        noise = generate_eeg_like_noise(initial_signal.shape, label_tensor=initial_labels, fs=256, device=self.device, dtype=self.unet.dtype)


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