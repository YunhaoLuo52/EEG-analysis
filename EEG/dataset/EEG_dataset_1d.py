import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from ..registry import EEGDiffDR

@EEGDiffDR.register_module()
class EEGDataset1D(Dataset):
    def __init__(self, csv_path, sequence_length=2560):
        """
        Z-SCORE NORMALIZATION: Consistent implementation
        """
        self.csv_path = csv_path
        self.sequence_length = sequence_length
        
        # Load data
        data = pd.read_csv(csv_path, header=None)
        self.data = data.values.astype(np.float32)
        
        print(f"Dataset loaded: {self.data.shape}")
        print(f"Original data range: [{np.min(self.data):.4f}, {np.max(self.data):.4f}]")
        
        # Z-SCORE NORMALIZATION
        # Calculate global statistics across all samples
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        
        print(f"Global statistics - Mean: {self.mean:.4f}, Std: {self.std:.4f}")
        
        # Apply z-score normalization
        self.normalized_data = (self.data - self.mean) / (self.std + 1e-8)
        
        print(f"Normalized data range: [{np.min(self.normalized_data):.4f}, {np.max(self.normalized_data):.4f}]")
        print(f"Normalized mean: {np.mean(self.normalized_data):.6f}, std: {np.std(self.normalized_data):.6f}")
        
        self.num_samples = self.data.shape[0]
        
        # Store normalization method for consistency
        self.norm_method = "z_score"

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        # Get normalized sequence
        sequence = self.normalized_data[index, :]
        
        # Convert to tensor
        sequence = torch.from_numpy(sequence).float()
        
        # Reshape to [channels, sequence_length]
        sequence = sequence.reshape(1, -1)  # 1 channel
        
        # NO ADDITIONAL TRANSFORM - normalization is already done
        return (sequence,)
    
    def denormalize(self, normalized_data):
        """
        Reverse z-score normalization
        """
        denormalized = normalized_data * self.std + self.mean
        return denormalized


@EEGDiffDR.register_module()
class PredictionEEGDataset1D(Dataset):
    def __init__(self, csv_path, sequence_length=2560, prediction_length=1280):
        """
        Dataset for long-term prediction with z-score normalization
        """
        self.csv_path = csv_path
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
        # Load data
        data = pd.read_csv(csv_path, header=None)
        self.data = data.values.astype(np.float32)
        
        # Z-SCORE NORMALIZATION
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        
        # Apply z-score normalization
        self.normalized_data = (self.data - self.mean) / (self.std + 1e-8)
        
        self.num_samples = self.data.shape[0]
        self.norm_method = "z_score"

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        sequence = self.normalized_data[index, :]
        sequence = torch.from_numpy(sequence).float()
        sequence = sequence.reshape(1, -1)
        return (sequence,)
    
    def denormalize(self, normalized_data):
        """
        Reverse z-score normalization
        """
        denormalized = normalized_data * self.std + self.mean
        return denormalized


@EEGDiffDR.register_module()
class EvaluationDataset1D(Dataset):
    def __init__(self, csv_path, window_size=2560, prediction_point=1280):
        """
        Evaluation dataset with z-score normalization
        """
        self.csv_path = csv_path
        self.window_size = window_size
        self.prediction_point = prediction_point
        
        # Load data
        data = pd.read_csv(csv_path, header=None)
        self.data = data.values.astype(np.float32)
        
        # Z-SCORE NORMALIZATION
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        
        # Apply z-score normalization
        self.normalized_data = (self.data - self.mean) / (self.std + 1e-8)
        
        self.num_samples = self.data.shape[0]
        self.norm_method = "z_score"

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        window = self.normalized_data[index, :]
        window = torch.from_numpy(window).float()
        window = window.reshape(1, -1)
        return (window,)
    
    def denormalize(self, normalized_data):
        """
        Reverse z-score normalization
        """
        denormalized = normalized_data * self.std + self.mean
        return denormalized