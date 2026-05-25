# TODO: fix issue regarding relative imports

from ...helpers.models import ModelConfig
from ..ml_models.vq_vae import VQVAE
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm

from helpers.data_tool import CMSDataTool, \
                                CMSPlots, \
                                PixelPatchesDataset

# Goal for the VQVAE postprocessing 
# is to create helper classes that could 
# be used in integrating to pileup_ml project

# Develop main VQVAE class for the pileup ml project

class CMSVQVAE:
    
    """
    Model for interfacing with VQ-VAE model
    """
    
    def __init__(
        self, 
        config: ModelConfig, 
        dataset: PixelPatchesDataset
    ) -> None:
        self.config = config
        
        self.vq = VQVAE(
            hidden_channels=self.config.hidden_channels, 
            kernel_size = self.config.kernel_size,
            slope=self.config.slope,
            n_res_layers=self.config.residual_layers, # residual hidden layers
            res_h_dim=self.config.residual_channels # residual hidden channels
        ).to(self.config.device)

        self.optimizer = optim.Adam(
            self.vq.parameters(), 
            lr=self.config.learning_rate, 
            amsgrad=True
        )
        
        self.dataset = dataset
        
        self.avg_loss_train = []
    
    # TODO: Add data loaders and continue wwriting training 
    # loop code
        
    def fit(self) -> None:
       start_time = datetime.now() 
       
       for epoch in range(self.config.num_epochs):
            self.vq.train()
            train_loss = 0
            
            progress_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f'Epoch {epoch+1}/{self.config.num_epochs}'
            )

       
       
       
       self.total_execution = (datetime.now() - start_time).seconds
        
       print(f"Training: {self.total_execution} seconds")