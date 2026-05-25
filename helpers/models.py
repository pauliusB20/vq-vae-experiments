from dataclasses import dataclass
import numpy as np

@dataclass      
class ModelConfig:
    
    """
    Model config class
    """
    
    batch_size: int
    beta: np.float64
    learning_rate: np.float64
    model_param_scale: int
    num_epochs: int
    channels: int
    hidden_channels: int
    num_embeddings: int
    latent_dim: int
    residual_channels: int
    residual_layers: int