import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe (e.g. running under tmux with no display)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from pydantic.dataclasses import dataclass as pydantic_dataclass
from torch.utils.data import Dataset
from torchvision import transforms

from pileup_ml.detectors.pixels import PixelDetector, PixelModule
from pileup_ml.pixels.hits import PixelDigiEvent
from pileup_ml.pixels.patches import event_hits_to_patches

from pythae.data.datasets import DatasetOutput
from pythae.models import VQVAE, VQVAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.pipelines.training import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
from sklearn.metrics import mean_squared_error 

load_dotenv()

# TEST_ITERATIONS = 10
BIT_DEPTHS = [8, 12, 16, 20, 24]
PIXEL_SIZES = [8, 12, 16, 20]
# BIT_DEPTHS = [8]
# PIXEL_SIZES = [8]
CHANNELS = 1
EVENT_COUNT_TRAIN = 3
EVENT_COUNT_TEST = 80
ADC_MAX = 255
SEED = 123
OUTPUT_DIR = "vqvae_pythae_pixels"
PLOT_PATH = "vqvae_pixels_reconstruction_rmse_vs_bit_depth.png"

KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1
ENCODED_PATCH_INDEXES = 4
EPOCHS = 100
BATCH_SIZE = 2**13


def codebook_size(bit_depth: int) -> int:
    # NOTE: kept from the original code. For bit_depth=8 this is only 4 codes.
    # Change this if you meant something else (e.g. 2**bit_depth).
    return 2 ** (bit_depth // 4)


@pydantic_dataclass
class CustomVQVAEConfig(VQVAEConfig):
    """pythae VQVAEConfig + the extra field needed for the Conv2d encoder/decoder."""
    hidden_channels: int = 16


class ResBlock2d(nn.Module):
    """Port of `ResBlock` from vqvae_layers.py, using Conv2d for 2D pixel patches."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class CustomEncoderConv(BaseEncoder):
    """Port of `EncoderConv`, flattened to a 2D embedding so pythae's stock
    `VQVAE` / `QuantizerEMA` can be used as-is.

    NOTE (fix vs. the version you pasted): pooling/residual here use the 2D
    (`AdaptiveAvgPool2d`) variants, since a Conv2d stack produces a 4D
    `(N, C, H, W)` feature map -- `AdaptiveAvgPool1d`/`ResBlock1d`-style 1D
    ops (as used in the strips script) can't operate on that.
    """

    def __init__(self, model_config: CustomVQVAEConfig) -> None:
        BaseEncoder.__init__(self)
        channels = model_config.input_dim[0]
        hidden_channels = model_config.hidden_channels
        latent_dim = model_config.latent_dim

        self.latent_dim = latent_dim
        self.encoded_patch_indexes = ENCODED_PATCH_INDEXES

        self.model = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=KERNEL_SIZE,
                      stride=STRIDE, padding=PADDING),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, latent_dim, kernel_size=KERNEL_SIZE - 1,
                      padding=PADDING),
        )
        self.adaptive_avg_2d = nn.AdaptiveAvgPool2d((ENCODED_PATCH_INDEXES, ENCODED_PATCH_INDEXES))
        self.residual = nn.Sequential(
            ResBlock2d(latent_dim, latent_dim // 2),
            ResBlock2d(latent_dim, latent_dim // 2),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        out = self.model(x)                 # (N, latent_dim, H', W')
        out = self.adaptive_avg_2d(out)      # (N, latent_dim, ENCODED_PATCH_INDEXES, ENCODED_PATCH_INDEXES)
        out = self.residual(out)
        out = out.reshape(out.shape[0], -1)
        return ModelOutput(embedding=out)


class CustomDecoderConv(BaseDecoder):
    """Port of `DecoderConv`, unflattening the 2D quantized vector from pythae's VQVAE.

    Outputs raw logits (no final sigmoid) -- reconstruction loss is computed
    with `F.binary_cross_entropy_with_logits` (see `CustomVQVAE._get_vae_loss`
    below). Apply `torch.sigmoid()` to `recon_x` yourself wherever you need an
    actual [0, 1] reconstruction (e.g. `reconstruct()` below).
    """

    def __init__(self, model_config: CustomVQVAEConfig) -> None:
        BaseDecoder.__init__(self)
        channels = model_config.input_dim[0]
        hidden_channels = model_config.hidden_channels
        latent_dim = model_config.latent_dim

        self.latent_dim = latent_dim
        self.encoded_patch_indexes = ENCODED_PATCH_INDEXES
        self.patch_size = model_config.input_dim[-1]

        self.residual = nn.Sequential(
            ResBlock2d(latent_dim, latent_dim // 2),
            ResBlock2d(latent_dim, latent_dim // 2),
            nn.ReLU(),
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_channels, kernel_size=KERNEL_SIZE - 1,
                               padding=PADDING),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, channels, kernel_size=KERNEL_SIZE,
                               stride=STRIDE, padding=PADDING),
        )

    def forward(self, z: torch.Tensor) -> ModelOutput:
        out = z.reshape(
            z.shape[0], self.latent_dim, self.encoded_patch_indexes, self.encoded_patch_indexes
        )
        out = self.residual(out)
        out = self.model(out)               # (N, channels, H_out, W_out)
        # Resize to the real patch size (no-op when it already matches).
        if out.shape[-2:] != (self.patch_size, self.patch_size):
            out = F.interpolate(
                out, size=(self.patch_size, self.patch_size), mode="bilinear",
                align_corners=False,
            )
        # NOTE: raw logits returned here -- no sigmoid. See class docstring.
        return ModelOutput(reconstruction=out)


class CustomVQVAE(VQVAE):
    """pythae's VQVAE with its default MSE reconstruction loss swapped for
    BCE-with-logits. Everything else (encoder, quantizer selection via
    `_set_quantizer`, `forward`) is untouched -- only `loss_function` is
    overridden.
    """

    def _get_vae_loss(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        VAE binary cross entropy loss
        """
        recon_loss = F.binary_cross_entropy_with_logits(
            recon_x,
            x,
            reduction="mean"
        )

        return recon_loss

    def loss_function(self, recon_x, x, quantizer_output):
        recon_loss = self._get_vae_loss(recon_x, x)

        vq_loss = quantizer_output.loss

        return (
            recon_loss + vq_loss.mean(dim=0),
            recon_loss,
            vq_loss.mean(dim=0),
        )


class AddNormalization:
    """Normalizes ADC data to [0, 1]."""

    def __call__(self, x) -> torch.Tensor:
        x = torch.as_tensor(x)
        return (x / ADC_MAX).float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ()"


class PixelsDataset(Dataset):
    """
    Pytorch dataset class for making PixelEventHit adcs into tensors.

    NOTE (fixes vs. the version you pasted):
    - `__getitem__` now returns a `DatasetOutput(data=...)` (pythae's own
      `OrderedDict`-with-attribute-access type), not a bare Tensor/ndarray --
      pythae's `TrainingPipeline` requires this once you hand it your own
      `Dataset` subclass (see `pythae/data/datasets.py`'s own `BaseDataset`).
    - Each patch now gets an explicit channel dimension
      (`(1, patch_size, patch_size)`), since Conv2d expects `(N, C, H, W)`.

    Returns:
        DatasetOutput: dict-like object with a 'data' key holding the patch
        as a torch.Tensor of shape (1, patch_size, patch_size)
    """

    def __init__(self, patches: np.ndarray, patch_size: int, transform=None):
        self.event_patches_adcs = [
            patch.reshape(patch_size, patch_size)
            for patch in patches
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.event_patches_adcs)

    def __getitem__(self, index: int) -> DatasetOutput:
        event_patch = self.event_patches_adcs[index]

        # uint16 -> int32: older torch versions cannot convert uint16 arrays
        event_patch = torch.as_tensor(event_patch.astype(np.int32)).unsqueeze(0)

        # Applying the transform
        if self.transform:
            event_patch = self.transform(event_patch)

        return DatasetOutput(data=event_patch)


class Helper:

    @staticmethod
    def hits_to_blocks(event: PixelDigiEvent, patches_size: int) -> np.ndarray:
        return event_hits_to_patches(
            event,
            segment_size=patches_size,
            fill_value=0,
            dtype=np.uint16,
        ).as_array()

    @staticmethod
    def get_patches_sets(
        pixels_events_train: list[PixelDigiEvent],
        pixels_events_test: list[PixelDigiEvent],
        patch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        train = np.concatenate(
            [Helper.hits_to_blocks(e, patch_size) for e in pixels_events_train]
        )
        test = np.concatenate(
            [Helper.hits_to_blocks(e, patch_size) for e in pixels_events_test]
        )
        return train, test


@torch.no_grad()
def reconstruct(model, data: torch.Tensor, device, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    outputs = []
    for start in range(0, data.shape[0], batch_size):
        batch = data[start:start + batch_size].to(device)
        out = model({"data": batch})
        recon = torch.sigmoid(out.recon_x)  # logits -> [0, 1] reconstruction
        outputs.append(recon.cpu())
    return torch.cat(outputs, dim=0)


def rmse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Mean over samples of the per-sample RMSE, in ADC counts."""
    original = np.round((original * ADC_MAX).cpu().numpy())
    reconstructed = np.round((reconstructed * ADC_MAX).cpu().numpy())

    per_sample_rmse = mean_squared_error(
        original,
        reconstructed
    )
    return float(per_sample_rmse**0.5)


def dataset_to_tensor(dataset: Dataset) -> torch.Tensor:
    return torch.stack([dataset[i]["data"] for i in range(len(dataset))])  # (N, 1, H, W)


def plot_rmse(rmse_values: dict, bit_depths: list[int], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for pixel_size, values in sorted(rmse_values.items()):
        ax.plot(bit_depths, values, marker="o", label=f"Pixel patch size {pixel_size}")
    ax.set_xlabel("Bit depth")
    ax.set_ylabel("RMSE (ADC)")
    ax.set_title("VQ-VAE RMSE by pixel patch size")
    ax.set_xticks(bit_depths)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Starting Pythae based custom VQ-VAE testing based on Pixel data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Loading pixel data")
    pixel_modules = PixelModule.read_json(
        '/data/cern/pileup_ml/detid_info/detids_bpix.json',
        '/data/cern/pileup_ml/detid_info/detids_fpix.json',
    )
    pixel_detector = PixelDetector(pixel_modules)

    # NOTE (fix): keep the raw event lists around unchanged across iterations.
    # The version you pasted reassigned `pixels_events_train`/`_test` with the
    # *patch arrays* returned by `get_patches_sets` inside the loop, so on the
    # second `pixel_size` the "events" being patchified were actually already
    # the previous iteration's patches, not the original digi events.
    pixels_events_train = PixelDigiEvent.read_root(
        "/data/cern/pileup_ml/premixlib2024/0001.root", detector=pixel_detector
    )[:EVENT_COUNT_TRAIN]
    pixels_events_test = PixelDigiEvent.read_root(
        "/data/cern/pileup_ml/premixlib2024/0002.root", detector=pixel_detector
    )[:EVENT_COUNT_TEST]

    data_transform = transforms.Compose([AddNormalization()])
    rmse_values = defaultdict(list)

    for pixel_size in PIXEL_SIZES:
        pixel_patches_train, pixel_patches_test = Helper.get_patches_sets(
            pixels_events_train, pixels_events_test, pixel_size
        )
        # Rebuild the data for every patch size
        print(f"Preparing train and test data for pixel patch size {pixel_size}")
        vqvae_trainset = PixelsDataset(pixel_patches_train, pixel_size, data_transform)
        vqvae_testset = PixelsDataset(pixel_patches_test, pixel_size, data_transform)
        test_data = dataset_to_tensor(vqvae_testset)

        for bit_depth in BIT_DEPTHS:
            start_time = datetime.now()
            print(f"Starting VQ-VAE pixels at {start_time:%Y-%m-%d %H:%M:%S}")

            num_embeddings = codebook_size(bit_depth)
            print(f"Applying: num_embeddings = {num_embeddings} "
                  f"(bit_depth={bit_depth}), pixel_size = {pixel_size}")

            # Fresh seed, config, encoder and decoder for every run so runs are independent
            torch.manual_seed(SEED)
            np.random.seed(SEED)

            model_config = CustomVQVAEConfig(
                input_dim=(CHANNELS, pixel_size, pixel_size),
                latent_dim=32,
                hidden_channels=16,
                num_embeddings=num_embeddings,
                use_ema=True,
                decay=0.9,
                commitment_loss_factor=0.01,
            )
            encoder = CustomEncoderConv(model_config)
            decoder = CustomDecoderConv(model_config)

            # CustomVQVAE = pythae's VQVAE + BCE-with-logits reconstruction loss
            # (instead of the default MSE).
            model = CustomVQVAE(
                model_config=model_config,
                encoder=encoder,
                decoder=decoder,
            ).to(device)

            print(f"embedding_dim resolved by pythae: {model.model_config.embedding_dim}")
            print(f"quantizer: {type(model.quantizer).__name__}")

            training_config = BaseTrainerConfig(
                output_dir=os.path.join(OUTPUT_DIR, f"pixels{pixel_size}_bd{bit_depth}"),
                learning_rate=1e-3,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                num_epochs=EPOCHS,
            )

            pipeline = TrainingPipeline(training_config=training_config, model=model)
            pipeline(train_data=vqvae_trainset)

            # Use the in-memory trained model (avoids reloading from the wrong folder
            # and AutoModel dropping the custom `hidden_channels` config field).
            reconstructions = reconstruct(model, test_data, device)
            print(reconstructions.shape)

            overall_rmse = rmse(test_data, reconstructions)
            rmse_values[pixel_size].append(overall_rmse)

            print(f"RMSE (eval set): {overall_rmse:.2f} ADC")

            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"Finished at {datetime.now():%Y-%m-%d %H:%M:%S}")
            print(f"Time difference = {elapsed:.0f} seconds\n------------------")

    # Plot once, after all pixel sizes and bit depths have finished
    print("Plotting overall patch based on different bit depth pixel VQ-VAE model rmse")
    plot_rmse(rmse_values, BIT_DEPTHS, PLOT_PATH)
    print(f"Saved plot to {PLOT_PATH}")
    print("DONE")


if __name__ == "__main__":
    main()