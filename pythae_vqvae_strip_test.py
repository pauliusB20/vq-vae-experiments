import os
from collections import defaultdict
from dataclasses import replace
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

from pileup_ml.compression.metrics import compare_hit_presence, compare_matching_hits
from pileup_ml.detectors.strips import StripsDetector
from pileup_ml.strips.hits import StripDigiEvent
from pileup_ml.strips.segments import event_hits_to_segments, event_segments_to_hits

from pythae.data.datasets import DatasetOutput
from pythae.models import VQVAE, VQVAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.pipelines.training import TrainingPipeline
from pythae.trainers import BaseTrainerConfig

load_dotenv()

# TEST_ITERATIONS = 10
BIT_DEPTHS = [8, 12, 16, 20, 24]
SEGMENT_SIZES = [8, 12, 16, 20]
# BIT_DEPTHS = [8]
# SEGMENT_SIZES = [8]
CHANNELS = 1
EVENT_COUNT_TRAIN = 3
EVENT_COUNT_TEST = 80
ADC_MAX = 1023
SEED = 123
OUTPUT_DIR = "my_custom_vqvae_model"
PLOT_PATH = "vqvae_strips_rmse_adc_vs_bit_depth_by_segment_size.png"
PRECISION_PLOT_PATH = "vqvae_strips_precision_vs_bit_depth_by_segment_size.png"
RECALL_PLOT_PATH = "vqvae_strips_recall_vs_bit_depth_by_segment_size.png"

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
    """pythae VQVAEConfig + the extra field needed for the Conv1d encoder/decoder."""
    hidden_channels: int = 16


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
    """Port of `EncoderConv`, flattened to a 2D embedding so pythae's stock
    `VQVAE` / `QuantizerEMA` can be used as-is."""

    def __init__(self, model_config: CustomVQVAEConfig) -> None:
        BaseEncoder.__init__(self)
        channels = model_config.input_dim[0]
        hidden_channels = model_config.hidden_channels
        latent_dim = model_config.latent_dim

        self.latent_dim = latent_dim
        self.encoded_patch_indexes = ENCODED_PATCH_INDEXES

        self.model = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=KERNEL_SIZE,
                      stride=STRIDE, padding=PADDING),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, latent_dim, kernel_size=KERNEL_SIZE - 1,
                      padding=PADDING),
        )
        self.adaptive_avg_1d = nn.AdaptiveAvgPool1d(ENCODED_PATCH_INDEXES)
        self.residual = nn.Sequential(
            ResBlock1d(latent_dim, latent_dim // 2),
            ResBlock1d(latent_dim, latent_dim // 2),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        out = self.model(x)                 # (N, latent_dim, L')
        out = self.adaptive_avg_1d(out)     # (N, latent_dim, ENCODED_PATCH_INDEXES)
        out = self.residual(out)
        out = out.reshape(out.shape[0], -1)
        return ModelOutput(embedding=out)


class CustomDecoderConv(BaseDecoder):
    """Port of `DecoderConv`, unflattening the 2D quantized vector from pythae's VQVAE.

    Outputs raw logits (no final sigmoid) -- reconstruction loss is computed
    with `F.binary_cross_entropy_with_logits`, which is numerically more
    stable than applying `sigmoid()` here and then plain BCE (see
    `CustomVQVAE._get_vae_loss` below). Apply `torch.sigmoid()` to
    `recon_x` yourself wherever you need an actual [0, 1] reconstruction
    (e.g. `reconstruct()` below).
    """

    def __init__(self, model_config: CustomVQVAEConfig) -> None:
        BaseDecoder.__init__(self)
        channels = model_config.input_dim[0]
        hidden_channels = model_config.hidden_channels
        latent_dim = model_config.latent_dim

        self.latent_dim = latent_dim
        self.encoded_patch_indexes = ENCODED_PATCH_INDEXES
        self.segment_size = model_config.input_dim[-1]

        self.residual = nn.Sequential(
            ResBlock1d(latent_dim, latent_dim // 2),
            ResBlock1d(latent_dim, latent_dim // 2),
            nn.ReLU(),
        )
        self.model = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden_channels, kernel_size=KERNEL_SIZE - 1,
                               padding=PADDING),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(hidden_channels, channels, kernel_size=KERNEL_SIZE,
                               stride=STRIDE, padding=PADDING),
        )

    def forward(self, z: torch.Tensor) -> ModelOutput:
        out = z.reshape(z.shape[0], self.latent_dim, self.encoded_patch_indexes)
        out = self.residual(out)
        out = self.model(out)               # length 8 for the default kernel/stride
        # The transposed convs always produce length 8, so resize to the real
        # segment size (no-op when segment_size == 8).
        if out.shape[-1] != self.segment_size:
            out = F.interpolate(out, size=self.segment_size, mode="linear",
                                align_corners=False)
        # NOTE: raw logits returned here -- no sigmoid. See class docstring.
        return ModelOutput(reconstruction=out)


class CustomVQVAE(VQVAE):
    """pythae's VQVAE with its default MSE reconstruction loss swapped for
    BCE-with-logits, so this baseline's loss matches the main model's (also
    EMA + BCE-with-logits). Everything else (encoder, quantizer selection via
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


class StripSegmentsDataset(Dataset):
    """Torch dataset turning strip segments into (1, segment_size) tensors."""

    def __init__(self, segments: np.ndarray, transform=None):
        self.segments = segments
        self.transform = transform

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> DatasetOutput:
        # uint16 -> int32: older torch versions cannot convert uint16 arrays
        segment = torch.as_tensor(self.segments[index].astype(np.int32))
        segment = segment.unsqueeze(0)
        if self.transform:
            segment = self.transform(segment)
        return DatasetOutput(data=segment)


def get_train_segments(events: list[StripDigiEvent], segment_size: int) -> np.ndarray:
    """All non-empty segments of the training events as one (N, segment_size) array."""
    return np.concatenate([
        event_hits_to_segments(e, segment_size=segment_size, fill_value=0,
                               dtype=np.uint16).as_array()
        for e in events
    ])


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


def reconstruct_event(model, event: StripDigiEvent, segment_size: int, device) -> StripDigiEvent:
    """Round-trip one event: hits -> segments -> VQ-VAE -> segments -> hits."""
    blocks = event_hits_to_segments(event, segment_size=segment_size, fill_value=0,
                                    dtype=np.uint16)
    if len(blocks) == 0:
        return event

    x = AddNormalization()(torch.as_tensor(blocks.as_array().astype(np.int32)))
    recon = reconstruct(model, x.unsqueeze(1), device).squeeze(1).numpy() * ADC_MAX

    # Same decoding as StripCompression.vectors_to_blocks: round to ADC counts,
    # and a strip is a hit only if its rounded ADC is > 0.
    adcs = np.round(recon).clip(0, ADC_MAX).astype(np.uint16)
    blocks = replace(blocks, adcs=adcs, occupancy=adcs > 0)
    return event_segments_to_hits(blocks)


def evaluate(model, events: list[StripDigiEvent], segment_size: int, device) -> dict[str, float]:
    """RMSE over matching hits plus hit-presence precision/recall (pileup_ml metrics)."""
    reconstructed = [reconstruct_event(model, e, segment_size, device) for e in events]

    precision, recall = compare_hit_presence(events, reconstructed)
    try:
        rmse = compare_matching_hits(events, reconstructed, metric="rmse")
    except ValueError:  # no strip is a hit in both original and reconstruction
        rmse = float("nan")
    return {"rmse": rmse, "precision": precision, "recall": recall}


def plot_metric(values: dict, bit_depths: list[int], ylabel: str, title: str,
                out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for segment, vals in sorted(values.items()):
        ax.plot(bit_depths, vals, marker="o", label=f"Segment size {segment}")
    ax.set_xlabel("Bit depth")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(bit_depths)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Starting Pythae based custom VQ-VAE testing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Loading strip data")
    detector_info = Path(os.environ["DETID_INFO_DIR"])
    root_dir = Path(os.environ["STRIP_ROOT_FILE_DIR"])
    strip_detector = StripsDetector.load(detector_info)
    strip_events_train = StripDigiEvent.read_root(
        root_dir / "0001_10.root", detector=strip_detector
    )[:EVENT_COUNT_TRAIN]
    strip_events_test = StripDigiEvent.read_root(
        root_dir / "0002_100.root", detector=strip_detector
    )[:EVENT_COUNT_TEST]

    data_transform = transforms.Compose([AddNormalization()])
    metric_values = {name: defaultdict(list) for name in ("rmse", "precision", "recall")}

    for segment in SEGMENT_SIZES:
        # Rebuild the training data for every segment size
        print(f"Preparing train data for segment size {segment}")
        vqvae_trainset = StripSegmentsDataset(
            get_train_segments(strip_events_train, segment), data_transform
        )

        for bit_depth in BIT_DEPTHS:
            start_time = datetime.now()
            print(f"Starting VQ-VAE strips at {start_time:%Y-%m-%d %H:%M:%S}")

            num_embeddings = codebook_size(bit_depth)
            print(f"Applying: num_embeddings = {num_embeddings} "
                  f"(bit_depth={bit_depth}), segment_size = {segment}")

            # Fresh seed, config, encoder and decoder for every run so runs are independent
            torch.manual_seed(SEED)
            np.random.seed(SEED)

            model_config = CustomVQVAEConfig(
                input_dim=(CHANNELS, segment),
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
            # (instead of the default MSE), so this baseline's loss matches the
            # main model's loss.
            model = CustomVQVAE(
                model_config=model_config,
                encoder=encoder,
                decoder=decoder,
            ).to(device)

            print(f"embedding_dim resolved by pythae: {model.model_config.embedding_dim}")
            print(f"quantizer: {type(model.quantizer).__name__}")

            training_config = BaseTrainerConfig(
                output_dir=os.path.join(OUTPUT_DIR, f"seg{segment}_bd{bit_depth}"),
                learning_rate=1e-3,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                num_epochs=EPOCHS,
            )

            pipeline = TrainingPipeline(training_config=training_config, model=model)
            pipeline(train_data=vqvae_trainset)

            # Use the in-memory trained model (avoids reloading from the wrong folder
            # and AutoModel dropping the custom `hidden_channels` config field).
            results = evaluate(model, strip_events_test, segment, device)
            for name, value in results.items():
                metric_values[name][segment].append(value)

            print(f"RMSE (matching hits): {results['rmse']:.2f} ADC | "
                  f"precision: {results['precision']:.4f} | recall: {results['recall']:.4f}")

            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"Finished at {datetime.now():%Y-%m-%d %H:%M:%S}")
            print(f"Time difference = {elapsed:.0f} seconds\n------------------")

    # Plot once, after all segment sizes and bit depths have finished
    print("Plotting metrics")
    plot_metric(metric_values["rmse"], BIT_DEPTHS, "RMSE (ADC)",
                "VQ-VAE RMSE by segment size", PLOT_PATH)
    plot_metric(metric_values["precision"], BIT_DEPTHS, "Precision",
                "VQ-VAE hit precision by segment size", PRECISION_PLOT_PATH)
    plot_metric(metric_values["recall"], BIT_DEPTHS, "Recall",
                "VQ-VAE hit recall by segment size", RECALL_PLOT_PATH)
    print(f"Saved plots to {PLOT_PATH}, {PRECISION_PLOT_PATH}, {RECALL_PLOT_PATH}")
    print("DONE")


if __name__ == "__main__":
    main()
