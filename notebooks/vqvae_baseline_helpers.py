from pythae.models.nn.benchmarks.mnist.resnets import Encoder_ResNet_VQVAE_MNIST, Decoder_ResNet_VQVAE_MNIST
from pythae.models import VQVAE, VQVAEConfig
from torch.nn import ModuleList,\
    Conv2d,\
        Sequential,\
            Module,\
                ReLU,\
                ConvTranspose2d
from torch import nn, tensor, Tensor
from collections import OrderedDict

class ModelOutput(OrderedDict):
    """Base ModelOutput class fixing the output type from the models. This class is inspired from
    the ``ModelOutput`` class from hugginface transformers library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> tuple:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())

class ResBlock(Module):
    def __init__(self, in_channels, out_channels):
        Module.__init__(self)

        self.conv_block = Sequential(
            ReLU(),
            Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: tensor) -> Tensor:
        return x + self.conv_block(x)

class Encoder_VQVAE_PATCH(Encoder_ResNet_VQVAE_MNIST):
    
    def __init__(self, args: VQVAEConfig):
        super().__init__(args)

        self.input_dim = (
            1, 
            args.patch_size, 
            args.patch_size
        )
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = ModuleList()

        layers.append(Sequential(
        Conv2d(
                in_channels=self.n_channels,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            )
        ))

        layers.append(Sequential(
            Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1
            )
        ))

        layers.append(Sequential(
            Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0
            )
        ))

        layers.append(
            Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.pre_qantized = Conv2d(128, self.latent_dim, 1, 1)
        
    def forward(self, x: Tensor, output_layer_levels: list[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level."""
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x
        
        # print("Encoder start")
        for i in range(max_depth):
            out = self.layers[i](out)
            # print(out.shape)
            
            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.pre_qantized(out)
        return output
        
class Decoder_VQVAE_PATCH(Decoder_ResNet_VQVAE_MNIST):
    
    def __init__(self, args: VQVAEConfig):
        super().__init__(args)

        self.input_dim = (
            1, 
            args.patch_size, 
            args.patch_size
        )
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = ModuleList()

        layers.append(nn.ConvTranspose2d(
            in_channels=self.latent_dim,
            out_channels=128,
            kernel_size=1,
            stride=1
        )) 

        layers.append(nn.ConvTranspose2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )) 

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=128, 
                    out_channels=64, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1, 
                    output_padding=1
                ),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=64, 
                    out_channels=self.n_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    output_padding=0
                ),
                nn.Sigmoid(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: Tensor, output_layer_levels: list[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code
            under the key `reconstruction`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `reconstruction_layer_i`
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})"
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z

        # print("Decoder")
        for i in range(max_depth):
            out = self.layers[i](out)
            # print(out.shape)
            
            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output
        
        