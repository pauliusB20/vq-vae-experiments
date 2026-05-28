# TODO: fix issue regarding relative imports

from ...helpers.models import ModelConfig
from torch.utils.data import DataLoader
from ..ml_models.vq_vae import VQVAE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm
import torch, os
from torchmetrics.regression import MeanSquaredError
from helpers.model_tool import ModelTool
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
        train_loader: DataLoader,
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
        
        self.train_loader = train_loader
        self.avg_loss_train = []
        self.avg_mse_loss_train = []
        
        self.plots = CMSPlots()
        self.model_tool = ModelTool()
        self.mse = MeanSquaredError()
        
        # create helper folders
        if not os.path.exists(config.model_folder):
            os.mkdir(config.model_folder)
            print(f"{config.model_folder} folder for saving model states created!")
    
    # TODO: Add data loaders and continue wwriting training 
    # loop code
    
    def fit(self) -> None:
       start_time = datetime.now() 
       
       for epoch in range(self.config.num_epochs):
            self.vq.train()
            train_mse_loss = 0
            train_loss = 0
            
            for (batch_idx, X_adcs) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                X_adcs = X_adcs.float()
                X_adcs = X_adcs.to(self.config.device)
                
                (
                    recon_patch, 
                    _,
                    _,
                    perplexity,
                    vq_loss
                ) = self.vq(
                    X_adcs
                )
                
                bce_loss = self._get_vae_loss(recon_patch, X_adcs)
                total_vq_loss = bce_loss + vq_loss
                train_loss += total_vq_loss.item()
                train_mse_loss += self.mse(recon_patch, X_adcs).item()
                
                total_vq_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.vq.parameters(), 
                    max_norm=self.config.model_param_scale
                )
                
                self.optimizer.step()
            
            avg_loss_value = train_loss / len(self.train_loader)
            avg_mse_value = train_mse_loss / len(self.train_loader)
            self.avg_loss_train.append(avg_loss_value)
            self.avg_mse_loss_train.append(avg_mse_value)
       
       self.total_execution = (datetime.now() - start_time).seconds 
       print(f"Training completed with runtime: {self.total_execution} seconds")
       
       self.save_model()
       
       
    def _get_vae_loss(
        recon_x: torch.tensor, 
        x: torch.tensor
    ) -> float:
        recon_loss = F.binary_cross_entropy(
            recon_x, 
            x, 
            reduction="mean"
        )
        return recon_loss
    
    
    """
    Debug method for displaying gradients
    """
    def display_gradients(self) -> None:
        self.model_tool._display_gradients(self.vq)
    
    """
    Loading the model from a config defined file
    """
    
    def load_model(self) -> None:
        self.vq.load_state_dict(
            torch.load(self.config.model_source_path)
        )
        self.vq.eval()
        print(f"Model loaded from path: {self.config.model_source_path}")
    
    """
    Saving the model in config defined file
    """
    
    def save_model(self) -> None:
        torch.save(
            self.vq.state_dict(),
            self.config.model_source_path
        )        
        print(f"SUCCESS: Model saved in {self.config.model_source_path}")
    
    
    def transform(self, x: torch.tensor) -> tuple|None:
        with torch.no_grad():
            output = self.vq.encoder(x)
            (
                recon_loss, 
                z_q, 
                perplexity, 
                min_encodings, 
                min_encoding_indices, 
                codebook
            ) = self.vq.vq_layer(output)
            return recon_loss, z_q
        return None

    def inverse_transform(self, z_q: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            output = self.vq.decoder(z_q)
            return output
        return None
     
    """
    Generating model specific plot
    
    curve_type: vq loss | mse loss
    """    
    def plot_curve(self, curve_type: str) -> None:
        match curve_type:
            case "vq_loss":
                self.plots.plot_curve(
                    y_values=self.avg_loss_train,
                    x_values=list(range(self.config.num_epochs)),
                    y_title="BCE Average loss",
                    title="VQ VAE, BCE encoder average loss",
                    x_label="Epochs"
                )
            
            case "mse_loss":
                self.plots.plot_curve(
                    y_values=self.avg_mse_loss_train,
                    x_values=list(range(self.config.num_epochs)),
                    y_title="MSE Average loss",
                    title="VQ VAE, MSE encoder average loss",
                    x_label="Epochs"
                )
                
            case "":
                pass
            
            