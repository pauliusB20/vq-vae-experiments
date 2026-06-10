from pileup_ml.pixels.patches import event_hits_to_patches
from pileup_ml.pixels.hits import PixelDigiEvent
from matplotlib.gridspec import GridSpec
from torch.nn import Module
import matplotlib.pyplot as plt
import numpy as np
import torch


class CMSDataTool:
    
    """
    Helper class for analyzing data
    """
    BITS_HIT = 25
    PATCH_SIZE = 8
    
    def __init__(self) -> None:
        pass
    
    
    def _get_encoded_patch_bits(
            self,
            encoded_size,
            patch_size,
            module_rows=160,
            module_cols=416
        ):
    
        (
            width, 
            height, 
            vq_n_embed
        ) = encoded_size
        
        vq_bits = width * height * np.log2(vq_n_embed)
        
        # Patch address bits
        bits_patch_row = np.ceil(
            np.log2(
                np.ceil(module_rows / patch_size)
            )
        )
        bits_patch_col = np.ceil(
            np.log2(
                np.ceil(module_cols / patch_size)
            )
        )
        total_bits = bits_patch_row + bits_patch_col + vq_bits
        
        return total_bits
    
    def _get_total_hit_bits(self, total_hits: int) -> int:
        
        # Patch address bits
        return total_hits * self.BITS_HIT
    
    
    def _get_total_patch_bits(
            self, 
            patch_amount: int, 
            encoded_patch_size: tuple, 
            patch_size: int
        ) -> int:
        encoded_patch_bits = self._get_encoded_patch_bits(
            encoded_patch_size, 
            patch_size
        )
        return np.uint64(
            patch_amount * encoded_patch_bits
        )
        
    def _get_compression_ratio(
            self,
            n_hits_total: int, 
            patch_amount: int, 
            encoded_patch_size: tuple, 
            patch_size: int
        ) -> float:
        total_bits = self._get_total_hit_bits(n_hits_total)
        
        total_patch_bits = self._get_total_patch_bits(
            patch_amount, 
            encoded_patch_size, 
            patch_size
        )
        return total_bits / total_patch_bits

    def _get_patch_count(self, event: PixelDigiEvent) -> int:
        event_patches_adcs_flat = event_hits_to_patches(
            event, 
            patch_size=self.PATCH_SIZE
        ).as_array()
        return len(event_patches_adcs_flat)
    
class PixelPatchesDataset(torch.utils.data.Dataset):
    def __init__(self, event: PixelDigiEvent, patch_size, transform=None):
        event_patches_adcs_flat = event_hits_to_patches(
            event, 
            patch_size=patch_size
        ).as_array()
        self.event_patches_adcs = [
            event_patch_flat.reshape(patch_size, patch_size)
            for event_patch_flat in event_patches_adcs_flat
            if not (np.all(event_patch_flat == 0))
        ]
        
        self.transform = transform

    # Defining the length of the dataset
    def __len__(self) -> int:
        return len(self.event_patches_adcs)

    # Defining the method to get an item from the dataset
    def __getitem__(self, index: int) -> np.array:
        # print(self.event_patches_adcs[index])
        event_patch = self.event_patches_adcs[index]
        
        # Applying the transform
        if self.transform:
            event_patch = self.transform(event_patch)
        
        return event_patch