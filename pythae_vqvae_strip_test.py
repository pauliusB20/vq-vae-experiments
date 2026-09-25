
import os
import json
from pathlib import Path

# Load main pileupml packages and nn model
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torchvision import transforms
from pileup_ml.strips.hits import DETID_SIZE, StripDigiEvent
from pileup_ml.strips.segments import StripEventSegments, event_hits_to_segments, event_segments_to_hits


from pythae.data.datasets import DatasetOutput

from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models import VQVAE, VQVAEConfig, AutoModel
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pileup_ml.detectors.strips import StripsDetector
# Helper classes 
from pydantic.dataclasses import dataclass as pydantic_dataclass
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()


TEST_ITERATIONS = 10
CHANNELS = 1
STRIP_SEGMENT_SIZE = 8
EVENT_COUNT = 3



@pydantic_dataclass
class CustomVQVAEConfig(VQVAEConfig):
    """pythae VQVAEConfig + the one extra field needed to build the
    Conv1d encoder/decoder ported from vqvae_layers.py."""
    hidden_channels: int = 16
    
KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1
ENCODED_PATCH_INDEXES = 4   # matches `encoded_patch_indexes` in vq_spec / ModelConfig


class ResBlock1d(nn.Module):
    """Port of `ResBlock` from vqvae_layers.py, using Conv1d for 1D vector data."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class CustomEncoderConv(BaseEncoder):
    """Port of `EncoderConv` from vqvae_layers.py, flattened to a 2D
    embedding so pythae's stock `VQVAE` / `QuantizerEMA` can be used as-is."""

    def __init__(self, model_config: CustomVQVAEConfig) -> None:
        BaseEncoder.__init__(self)
        channels = model_config.input_dim[0]
        hidden_channels = model_config.hidden_channels
        latent_dim = model_config.latent_dim

        self.latent_dim = latent_dim
        self.encoded_patch_indexes = ENCODED_PATCH_INDEXES

        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=hidden_channels,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=latent_dim,
                kernel_size=KERNEL_SIZE - 1,
                padding=PADDING,
            ),
        )

        self.adaptive_avg_1d = nn.AdaptiveAvgPool1d(ENCODED_PATCH_INDEXES)

        self.residual = nn.Sequential(
            ResBlock1d(in_channels=latent_dim, out_channels=latent_dim // 2),
            ResBlock1d(in_channels=latent_dim, out_channels=latent_dim // 2),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        out = self.model(x)                     # (N, latent_dim, L)
        out = self.adaptive_avg_1d(out)          # (N, latent_dim, ENCODED_PATCH_INDEXES)
        out = self.residual(out)                 # (N, latent_dim, ENCODED_PATCH_INDEXES)
        out = out.reshape(out.shape[0], -1)       # (N, latent_dim * ENCODED_PATCH_INDEXES)
        return ModelOutput(embedding=out)


class CustomDecoderConv(BaseDecoder):
    """Port of `DecoderConv` from vqvae_layers.py, unflattening the 2D
    quantized vector handed back by pythae's stock `VQVAE.forward`."""

    def __init__(self, model_config: CustomVQVAEConfig) -> None:
        BaseDecoder.__init__(self)
        channels = model_config.input_dim[0]
        hidden_channels = model_config.hidden_channels
        latent_dim = model_config.latent_dim

        self.latent_dim = latent_dim
        self.encoded_patch_indexes = ENCODED_PATCH_INDEXES

        self.adaptive_avg_1d = nn.AdaptiveAvgPool1d(ENCODED_PATCH_INDEXES)

        self.residual = nn.Sequential(
            ResBlock1d(in_channels=latent_dim, out_channels=latent_dim // 2),
            ResBlock1d(in_channels=latent_dim, out_channels=latent_dim // 2),
            nn.ReLU(),
        )

        self.model = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=latent_dim,
                out_channels=hidden_channels,
                kernel_size=KERNEL_SIZE - 1,
                padding=PADDING,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(
                in_channels=hidden_channels,
                out_channels=channels,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING,
            ),
        )

    def forward(self, z: torch.Tensor) -> ModelOutput:
        out = z.reshape(z.shape[0], self.latent_dim, self.encoded_patch_indexes)
        out = self.adaptive_avg_1d(out)   # kept for parity with vqvae_layers.py; no-op here
        out = self.residual(out)
        out = self.model(out)             # (N, channels, segment_size)
        out = torch.sigmoid(out)          # reconstruction in [0, 1], since pythae's
                                           # default VQVAE.loss_function uses plain MSE
                                           # (not BCE-with-logits like vqvae_quantizer.py)
        return ModelOutput(reconstruction=out)
    
class AddNormalization:
    
    """
    Helper class for normalizing data from 0 to 1
    """
    
    def __call__(self, x: np.array) -> torch.Tensor:
        x = torch.as_tensor(x)
        x = x / 1023
        x = x.float()
        return x
    
    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name} ()"
    
class StripSegmentsDataset(Dataset):
    """
    Pytorch dataset class for making PixelEventHit adcs into tensors

    Returns:
        DatasetOutput: dict-like object with a 'data' key holding the
        segment as a torch.Tensor
    """

    def __init__(self, segments: list[object], transform=None):
        self.segments = segments
        self.transform = transform

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> DatasetOutput:

        event_segment = torch.as_tensor(self.segments[index])
        event_segment = event_segment.unsqueeze(0)
        
        # Applying the transform
        if self.transform:
            event_segment = self.transform(event_segment)

        return DatasetOutput(data=event_segment)   # <-- was: return event_segment


    
class Helper:
    
    @staticmethod
    def hits_to_vectors(event: StripDigiEvent) -> StripEventSegments:
        return event_hits_to_segments(
                event,
                segment_size=8,
                fill_value=0,
                dtype=np.uint16
        ).as_array()
        
    @staticmethod
    def _get_strips_sets(
        strip_events_train: list[StripDigiEvent], 
        strip_events_test: list[StripDigiEvent]
    ) -> StripEventSegments:
        vectors_collected_train = []      

        for event_train in strip_events_train:
            vectors = Helper.hits_to_vectors(event=event_train)
            vectors_collected_train.append(vectors)
        strip_segments_train = np.concatenate(vectors_collected_train)

        vectors_collected_test = []      
        for event_test in strip_events_test:
            vectors = Helper.hits_to_vectors(event_test)
            vectors_collected_test.append(vectors)
            
        strip_segments_test = np.concatenate(vectors_collected_test)
        return strip_segments_train, strip_segments_test


if __name__ == "__main__":
    
    print("Starting Pythae based custom VQ-VAE testing")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # For reproducible results (mirrors simple_vqvae_ema_strips.ipynb)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    print("Loading strip data")
    detector_info = Path(os.environ['DETID_INFO_DIR'])
    strip_detector = StripsDetector.load(detector_info)
    strip_events_train = StripDigiEvent.read_root(Path(os.environ['STRIP_ROOT_FILE_DIR']) / '0001_10.root', detector=strip_detector)[:EVENT_COUNT]
    strip_events_test = StripDigiEvent.read_root(Path(os.environ['STRIP_ROOT_FILE_DIR']) / '0002_100.root', detector=strip_detector)[:EVENT_COUNT]

    SEED = 123
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    print("Prearing train and test data")
    
    (
        strip_segments_train,
        strip_segments_test
    ) = Helper._get_strips_sets(
        strip_events_train,
        strip_events_test
    )

    data_transform = transforms.Compose([
        AddNormalization()
    ])
    vqvae_trainset = StripSegmentsDataset(strip_segments_train, data_transform)
    vqvae_testset = StripSegmentsDataset(strip_segments_test, data_transform)
    model_config = CustomVQVAEConfig(
        input_dim=(CHANNELS, STRIP_SEGMENT_SIZE),
        latent_dim=32,
        hidden_channels=16,
        num_embeddings=128,
        use_ema=True,
        decay=0.99,
        commitment_loss_factor=0.01,   # plays the role of `beta` in vqvae_quantizer.py
    )
    
    encoder = CustomEncoderConv(model_config)
    decoder = CustomDecoderConv(model_config)

    # Stock pythae VQVAE -- `_set_quantizer` sees a 2D encoder embedding and
    # automatically instantiates `QuantizerEMA` (since `use_ema=True`); no
    # subclassing / custom quantizer needed.
    
    rmse_values = []
    for iteration in range(TEST_ITERATIONS):
        
        start_time = datetime.now()
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Starting VQ-VAE strips at {start_time_str}")
        
        model = VQVAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
        ).to(device)
        
        print(f"embedding_dim resolved by pythae: {model.model_config.embedding_dim}")
        print(f"quantizer: {type(model.quantizer).__name__}")
        
        training_config = BaseTrainerConfig(
            output_dir="my_custom_vqvae_model",
            learning_rate=4e-3,
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            num_epochs=100,   # matches `num_epochs` in vq_spec
        )

        pipeline = TrainingPipeline(
            training_config=training_config,
            model=model,
        )
        
        pipeline(
            train_data=vqvae_trainset,
            eval_data=vqvae_testset,
        )
        
        last_training = sorted(os.listdir("vqvae"))[-1]
        final_model_dir = os.path.join("vqvae", last_training, "final_model")

        trained_model = AutoModel.load_from_folder(final_model_dir).to(device)
        trained_model.eval()
        
        @torch.no_grad()
        def reconstruct(model, data, batch_size=256):
            model.eval()
            outputs = []
            for start in range(0, data.shape[0], batch_size):
                batch = data[start:start + batch_size].to(device)
                out = model({"data": batch})
                outputs.append(out.recon_x.cpu())
            return torch.cat(outputs, dim=0)


        reconstructions = reconstruct(trained_model, vqvae_testset)
        print(reconstructions.shape)
        
        
        def rmse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
            """Root-mean-square error between original and reconstructed vectors."""
            diff = (original - reconstructed).reshape(original.shape[0], -1)
            per_sample_mse = torch.mean(diff ** 2, dim=1)
            return torch.sqrt(per_sample_mse).mean().item()


        overall_rmse = rmse(vqvae_testset, reconstructions)
        rmse_values.append(overall_rmse)
        
        print(f"RMSE (eval set): {overall_rmse:.6f}")
        
        time_diff_seconds = (datetime.now() - start_time).seconds
        end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Finsihed at {end_time_str}")
        print(f"Time difference seconds = {time_diff_seconds} seconds\n")
        
print("Plotting overall rmse")
steps = list(range(len(rmse_values)))

plt.figure(figsize=(8, 5))
plt.plot(steps, rmse_values, marker="o", linewidth=1.5, markersize=4)
plt.xlabel("Step")
plt.ylabel("RMSE")
plt.title("RMSE over training steps")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("vqvae_strips_rmse_over_steps.png", dpi=200, bbox_inches="tight")
# use .pdf or .svg instead of .png for a vector version, e.g.:
# plt.savefig("vqvae_strips_rmse_over_steps.pdf", bbox_inches="tight")

plt.show()
        
print("DONE")
        