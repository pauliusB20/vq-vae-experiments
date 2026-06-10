from torch.nn import Module
from torch import device
import numpy as np
import torch

class ModelTool:
    
    """
    Helper class for debuging 
    deep learning models
    """
    
    def __init__(self) -> None:
        pass
    
    def _display_gradients(self, model: Module) -> None:
        print("Showing model gradients")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(name, param.grad.norm().item())
                
    def _tensor_to_numpy(self, values: torch.Tensor) -> np.array:
        return values.cpu().detach().numpy()
    
    def set_seed(self, num: int, cuda: device = None) -> None:
        torch.manual_seed(num)
        # TODO: configure pytorch service for deterministic algorithms
        # torch.use_deterministic_algorithms(mode=True)
        if cuda:
            torch.cuda.manual_seed(num)
            torch.cuda.manual_seed_all(num)