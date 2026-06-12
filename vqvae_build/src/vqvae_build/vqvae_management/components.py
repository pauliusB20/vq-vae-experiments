# TODO: fix issue regarding relative imports

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm
import torch, os, sys
from torchmetrics.regression import MeanSquaredError

from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

# main component classes for running VQ-VAE
from vqvae_build.helpers.model_tool import ModelTool
from vqvae_build.helpers.data_tool import CMSPlots
from vqvae_build.helpers.models import ModelConfig
from vqvae_build.vqvae_model.networks import VQVAE

# Goal for the VQVAE postprocessing 
# is to create helper classes that could 
# be used in integrating to pileup_ml project

# Develop main VQVAE class for the pileup ml project

# TODO: Apply new fixes
class CMSVQVAE:
    
    """
    Model for interfacing with VQ-VAE model
    """
    
    def __init__(
        self, 
        config: ModelConfig
    ) -> None:
        self.config = config
        
        self.vq = VQVAE(
            channels=self.config.channels,
            hidden_channels=self.config.hidden_channels, 
            kernel_size = self.config.kernel_size,
            slope=self.config.slope,
            num_embeddings=self.config.num_embeddings,
            latent_dim=self.config.latent_dim,
            beta=self.config.beta,
            n_res_layers=self.config.residual_layers, # residual hidden layers
            res_h_dim=self.config.residual_channels, # residual hidden channels
            encoded_patch_indexes=self.config.encoded_patch_indexes,
            device=self.config.device
        ).to(self.config.device)

        self.optimizer = optim.Adam(
            self.vq.parameters(), 
            lr=self.config.learning_rate, 
            amsgrad=True
        )
        
        self.avg_loss_train = []
        self.avg_mse_loss_train = []
        self.avg_perplexity_train = []
        
        self.plots = CMSPlots()
        self.model_tool = ModelTool()
        self.mse = MeanSquaredError().to(config.device)
        device_seed = config.device
        self.model_tool.set_seed(config.seed, device_seed)
        
        # create helper folders
        if not os.path.exists(config.model_folder):
            os.mkdir(config.model_folder)
            print(f"{config.model_folder} folder for saving model states created!")

        
    
    def fit(
            self, 
            dataset: torch.utils.data.Dataset, 
            verbose: bool = False
        ) -> None:
        
        
       start_time = datetime.now() 
       self.avg_loss_train = []
       self.avg_mse_loss_train = []
       self.avg_perplexity_train = []
       
       train_loader = DataLoader(
            dataset=dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        ) 
       
       for epoch in range(self.config.num_epochs + 1):
            self.vq.train()
            train_perplexity = 0
            train_mse_loss = 0
            train_loss = 0
            
            for (_, X_adcs) in enumerate(train_loader):
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
                train_perplexity += perplexity.item()
                
                total_vq_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.vq.parameters(), 
                    max_norm=self.config.model_param_scale
                )
                
                self.optimizer.step()
            
            
            avg_loss_value = train_loss / len(train_loader)
            avg_mse_value = train_mse_loss / len(train_loader)
            avg_perplexity = train_perplexity / len(train_loader)
            
            self.avg_loss_train.append(avg_loss_value)
            self.avg_mse_loss_train.append(avg_mse_value)
            self.avg_perplexity_train.append(avg_perplexity)
            
            if verbose:
                print(
                    f"Epoch ({epoch}/{self.config.num_epochs})"
                    f" model VQ_LOSS + BCE Loss = {avg_loss_value},"
                    f" avg model codebook perplexity = {avg_perplexity}"
                )
       
       self.total_execution = (datetime.now() - start_time).seconds 
       print(f"Training completed with runtime: {self.total_execution} seconds")
       
       self.save_model()
     
     
    """
    VAE binary cross entropy loss 
    """         
    def _get_vae_loss(
        self,
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
        print(f"SUCCESS: Model loaded from path: {self.config.model_source_path}")
    
    """
    Saving the model in config defined file
    """
    def save_model(self) -> None:
        torch.save(
            self.vq.state_dict(),
            self.config.model_source_path
        )        
        print(f"SUCCESS: Model saved in {self.config.model_source_path}")
    
    """
    Patch compression
    """
    def transform(self, patch: torch.tensor) -> tuple|None:
        self.vq.eval()
        with torch.no_grad():
            output = self.vq.encoder(patch)
            (
                recon_loss, # debug 
                z_q, 
                perplexity, 
                min_encodings, # debug
                min_encoding_indices, # debug
                codebook # debug
            ) = self.vq.vq_layer(output)
        return perplexity, z_q

    """
    Patch reconstruction after compression
    """
    def inverse_transform(self, z_q: torch.tensor) -> torch.tensor:
        self.vq.eval()
        with torch.no_grad():
            output = self.vq.decoder(z_q)
        return output
     
    """
    Generating model specific plot
    
    curve_type: vq loss | mse loss | perplexity
    """    
    def plot_curve(self, curve_type: str) -> None:
        
        match curve_type:
            case "vq_loss":
                self.plots.plot_curve(
                    y_values=self.avg_loss_train,
                    x_values=list(range(self.config.num_epochs + 1)),
                    y_title="BCE Average loss",
                    title="VQ VAE, BCE encoder average loss",
                    x_label="Epochs"
                )
            
            case "mse_loss":
                self.plots.plot_curve(
                    y_values=self.avg_mse_loss_train,
                    x_values=list(range(self.config.num_epochs + 1)),
                    y_title="MSE Average loss",
                    title="VQ VAE, MSE encoder average loss",
                    x_label="Epochs"
                )
                
            case "perplexity":
                self.plots.plot_curve(
                    y_values=self.avg_perplexity_train,
                    x_values=list(range(self.config.num_epochs + 1)),
                    y_title="Perplexity Average",
                    title="VQ VAE, Perplexity average across epochs",
                    x_label="Epochs"
                )
                
            case _:
                raise Exception("Unsupported plot type!")
            
            