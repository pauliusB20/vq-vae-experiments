# Goal of Vector Quantization is to solve posterior collapse problem
import torch.nn.functional as F
from torch import device
from torch import nn
import numpy as np
import torch


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z: torch.tensor) -> tuple:
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        # NOTE: is a matrix learnable vector for learning codebook indices. 
        codebook = self.embedding.weight
        
        # Euclidean distance for finding closest codebooks
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, codebook.t())
            
        # Result -> d gets dimensions from matrix torch.matmul(z_flattened, self.embedding.weight.t())
        # embedding weight t is defined by num_embeddings, latent_dim
        # --------------------
            
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        
        # TODO: maybe need to have to(device)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e
        ).to(z.device)
        
        # perform one hot encoding
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # NOTE: selects weights form embedding weight and puts them based on one hot encoding
        z_q = torch.matmul(min_encodings, codebook).view(z.shape)
        
        # beta - commitment cost used in loss term
        
        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients with preservation trick
        z_q = z + (z_q - z).detach()

        # perplexity - average of weights of probabilities
        e_mean = torch.mean(min_encodings, dim=0)
        
        # Measures how many codebook vectors are beeing used
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        # contiguous() is used for improving performance
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return (
            loss, 
            z_q, 
            perplexity, 
            min_encodings, 
            min_encoding_indices, 
            codebook
        )
    
class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim, slope):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers, slope):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.slope = slope
        self.stack = nn.ModuleList(
            [
                ResidualLayer(in_dim, h_dim, res_h_dim, slope)
                for _ in range(n_res_layers)
            ])

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x
    
    
# Comments
# kernel_size = 4 - preserves spatial repesentations (coordinates)
class VQVAE(nn.Module):
    
    def __init__(
        self, 
        channels: int,
        latent_dim: int,
        num_embeddings: int,
        hidden_channels: int, 
        beta: np.float64,
        kernel_size: int, 
        slope: float,
        res_h_dim: int,
        n_res_layers: int,
        residual_slope: float = 0.08,
    ) -> None:
        super().__init__()
        
        self.encoder = nn.Sequential(   
            nn.Conv2d(
                channels, 
                hidden_channels, 
                kernel_size=kernel_size, 
                stride=2, 
                padding=1, 
                bias=False
            ),  
            nn.GroupNorm(
                num_groups=8, 
                num_channels=hidden_channels
            ),
            nn.LeakyReLU(slope),
            nn.Conv2d(
                hidden_channels, 
                hidden_channels * 2, 
                kernel_size=kernel_size, 
                stride=2, 
                padding=1, 
                bias=False
            ),
            nn.GroupNorm(
                num_groups=8, 
                num_channels=hidden_channels * 2
            ),
            nn.LeakyReLU(slope),            
            nn.Conv2d(
                hidden_channels * 2, 
                latent_dim, 
                kernel_size=kernel_size-1, 
                stride=1, 
                padding=1, 
                bias=False
            ),
            nn.GroupNorm(
                num_groups=8, 
                num_channels=latent_dim
            ),
            ResidualStack(
                latent_dim, 
                latent_dim, 
                res_h_dim, 
                n_res_layers, 
                residual_slope
            )
        )
        
        self.vq_layer = VectorQuantizer(
            num_embeddings, 
            latent_dim, 
            beta
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim, 
                hidden_channels * 2, 
                kernel_size=kernel_size - 1, 
                stride=1, 
                padding=1, 
                bias=False
            ),
            nn.GroupNorm(
                num_groups=8, 
                num_channels=hidden_channels * 2
            ),
            ResidualStack(
                hidden_channels * 2, 
                hidden_channels * 2, 
                res_h_dim, 
                n_res_layers, 
                residual_slope
            ),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.GroupNorm(
                num_groups=8, 
                num_channels=hidden_channels
            ),
            nn.LeakyReLU(slope),
            nn.ConvTranspose2d(
                hidden_channels, 
                channels, 
                kernel_size=kernel_size, 
                stride=2, 
                padding=1, 
                bias=False
            ),
            nn.Sigmoid()
        )

        
    def forward(self, x):
        z_encoded = self.encoder(x).to(x.device)
        
        (
            recon_loss, 
            z_q, 
            perplexity, 
            min_encodings, 
            min_encoding_indices, 
            codebook
        ) = self.vq_layer(z_encoded)
        
        output = self.decoder(z_q)
        
        return (
            output, 
            codebook, 
            z_q,
            perplexity, 
            recon_loss 
        )