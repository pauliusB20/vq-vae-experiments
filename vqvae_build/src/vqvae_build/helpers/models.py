from dataclasses import dataclass
from torch import device
import numpy as np
import os



@dataclass      
class ModelConfig:
    
    """
    Model config class
    """
    
    model_name: str
    model_folder: str
    batch_size: int
    beta: np.float64
    learning_rate: np.float64
    model_param_scale: int
    num_epochs: int
    channels: int
    hidden_channels: int
    num_embeddings: int
    kernel_size: int
    slope: np.float64
    latent_dim: int
    residual_channels: int
    residual_layers: int
    device: device
    seed: int
    encoded_patch_indexes: int
    
    @property
    def model_source_path(self) -> str:
        model_path = os.path.join(
            self.model_folder,
            self.model_name + ".pth"
        )
        return model_path