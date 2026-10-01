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

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

from pileup_ml.compression.metrics import compare_hit_presence, compare_matching_hits
from pileup_ml.detectors.pixels import PixelDetector, PixelModule
from pileup_ml.pixels.hits import PixelDigiEvent
# NOTE (assumption): `event_patches_to_hits` is guessed by analogy with the
# strips module's `event_hits_to_segments` / `event_segments_to_hits` pair --
# only the forward direction (`event_hits_to_patches`) had been imported
# here before. If the real reverse function has a different name in
# `pileup_ml.pixels.patches`, just swap it in this import line.
from pileup_ml.pixels.patches import event_hits_to_patches, event_patches_to_hits

from pythae.data.datasets import DatasetOutput
from pythae.models import VQVAE, VQVAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.pipelines.training import TrainingPipeline
from pythae.trainers import BaseTrainerConfig

load_dotenv()

# TEST_ITERATIONS = 10

BIT_DEPTHS = [8, 12, 16, 20, 24, 28, 32]

PIXEL_SIZES = [4, 6, 8, 10, 12, 14, 16]

# BIT_DEPTHS = [8]

# PIXEL_SIZES = [8]

CHANNELS = 1

EVENT_COUNT_TRAIN = 3

EVENT_COUNT_TEST = 80

ADC_MAX = 255

SEED = 123

OUTPUT_DIR = "vqvae_pythae_pixels"

PLOT_PATH = "vqvae_pixels_rmse_adc_vs_bit_depth_by_pixel_size.png"
PRECISION_PLOT_PATH = "vqvae_pixels_precision_vs_bit_depth_by_pixel_size.png"
RECALL_PLOT_PATH = "vqvae_pixels_recall_vs_bit_depth_by_pixel_size.png"
PERPLEXITY_PLOT_PATH = "vqvae_pixels_perplexity_vs_bit_depth_by_patch_size.png"

KERNEL_SIZE = 4

STRIDE = 2

PADDING = 1

ENCODED_PATCH_INDEXES = 2

EPOCHS = 100

BATCH_SIZE = 8096

# Which module to use for the per-iteration debug plot (arbitrary, matches
# the module index already used for the one-off debug plot in main()).
DEBUG_MODULE_INDEX = 29

LEARNING_RATE = 2e-3


def codebook_size(bit_depth: int) -> int:

    # NOTE: kept from the original code. For bit_depth=8 this is only 4 codes.

    # Change this if you meant something else (e.g. 2**bit_depth).

    return 2 ** (bit_depth // 4)


@pydantic_dataclass
class CustomVQVAEConfig(VQVAEConfig):
    """pythae VQVAEConfig + the extra fields needed for the Conv2d encoder/decoder.

    `encoded_patch_size` is the spatial size (H=W) right after the encoder's

    conv stack (before `adaptive_avg_2d` pools it down to

    `ENCODED_PATCH_INDEXES` x `ENCODED_PATCH_INDEXES`). It's computed by

    `CustomEncoderConv` and then written back onto the config so

    `CustomDecoderConv` can pool the quantized feature map back up to that

    exact size before its conv-transpose stack runs -- see `main()`.

    """

    hidden_channels: int = 16

    encoded_patch_size: int = 0


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
    """Port of `EncoderConv`. Precomputes the spatial size coming out of the

    conv stack (`encoded_patch_size`) *before* handing it to

    `adaptive_avg_2d`, so `CustomDecoderConv` can later pool the quantized

    feature map back up to that exact size before its own conv-transpose

    stack runs.

    NOTE on the "dummy dimension": the embedding here is returned as-is in

    (N, C, H, W) form -- a Conv2d stack already produces a genuine 4D

    tensor, unlike the 1D strips encoder (which needed a dummy `unsqueeze`

    to fake a 4th dimension for pythae's `_set_quantizer`/`forward` to treat

    it as a spatial grid). Adding an *extra* dummy dim on top of an already-4D

    tensor would make it 5D, which breaks pythae's `z.permute(0, 2, 3, 1)`

    (that call requires exactly 4 dims) -- so no dummy dim is added here.

    Dropping the `reshape(out.shape[0], -1)` flatten that used to be here is

    the actual analogue of the strips fix: it's what lets each of the

    ENCODED_PATCH_INDEXES x ENCODED_PATCH_INDEXES positions get its own

    independent codebook lookup, instead of collapsing the whole patch into

    a single code.

    """

    def __init__(self, model_config: CustomVQVAEConfig) -> None:

        BaseEncoder.__init__(self)

        channels = model_config.input_dim[0]

        hidden_channels = model_config.hidden_channels

        latent_dim = model_config.latent_dim

        patch_size = model_config.input_dim[-1]

        self.latent_dim = latent_dim

        self.encoded_patch_indexes = ENCODED_PATCH_INDEXES

        self.model = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                latent_dim,
                kernel_size=KERNEL_SIZE - 1,
                padding=PADDING,
            ),
        )

        # Precompute the spatial size (H=W) coming out of the conv stack

        # above (standard conv2d size formula, applied to both spatial dims

        # since kernels/strides/paddings here are symmetric), starting from

        # the real patch size, *before* adaptive_avg_2d pools it down to

        # ENCODED_PATCH_INDEXES x ENCODED_PATCH_INDEXES.

        encoded_patch_size = patch_size

        for layer in self.model:

            if isinstance(layer, nn.Conv2d):

                encoded_patch_size = (
                    (encoded_patch_size + 2 * layer.padding[0] - layer.kernel_size[0])
                    // layer.stride[0]
                ) + 1

        self.encoded_patch_size = encoded_patch_size

        self.adaptive_avg_2d = nn.AdaptiveAvgPool2d(
            (ENCODED_PATCH_INDEXES, ENCODED_PATCH_INDEXES)
        )

        self.residual = nn.Sequential(
            ResBlock2d(latent_dim, latent_dim // 2),
            ResBlock2d(latent_dim, latent_dim // 2),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:

        out = self.model(x)  # (N, latent_dim, encoded_patch_size, encoded_patch_size)

        out = self.residual(out)

        out = self.adaptive_avg_2d(
            out
        )  # (N, latent_dim, ENCODED_PATCH_INDEXES, ENCODED_PATCH_INDEXES)

        return ModelOutput(embedding=out)


class CustomDecoderConv(BaseDecoder):
    """Mirrors `CustomEncoderConv` in reverse: pools the quantized feature

    map from ENCODED_PATCH_INDEXES x ENCODED_PATCH_INDEXES back up to

    encoded_patch_size x encoded_patch_size (the size the encoder's conv

    stack produced, before *its* adaptive-pool step), runs it through the

    residual blocks, then upsamples back to the full patch size with the

    conv-transpose stack.

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

        self.adaptive_avg_2d = nn.AdaptiveAvgPool2d(
            (model_config.encoded_patch_size, model_config.encoded_patch_size)
        )

        self.residual = nn.Sequential(
            ResBlock2d(latent_dim, latent_dim // 2),
            ResBlock2d(latent_dim, latent_dim // 2),
            nn.ReLU(),
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim,
                hidden_channels,
                kernel_size=KERNEL_SIZE - 1,
                padding=PADDING,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_channels,
                channels,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING,
            ),
        )

    def forward(self, z: torch.Tensor) -> ModelOutput:

        out = self.adaptive_avg_2d(
            z
        )  # (N, latent_dim, encoded_patch_size, encoded_patch_size)

        out = self.residual(out)

        out = self.model(out)  # (N, channels, patch_size, patch_size)

        # No resize step: adaptive_avg_2d pools the quantized feature map
        # back up to exactly `encoded_patch_size` (the size the encoder's
        # conv stack produced before *its* pool), and the conv-transpose
        # stack mirrors the encoder's conv stack kernel-for-kernel, so its
        # output already lands on the real patch size exactly -- same as
        # the strips decoder, which has no interpolate/resize step either.

        # NOTE: raw logits returned here -- no sigmoid. See class docstring.

        return ModelOutput(reconstruction=out)


class CustomVQVAE(VQVAE):
    """pythae's VQVAE with its default MSE reconstruction loss swapped for

    BCE-with-logits. Everything else (encoder, quantizer selection via

    `_set_quantizer`, `forward`) is untouched -- only `loss_function` is

    overridden.

    """

    def _get_vae_loss(self, recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """

        VAE binary cross entropy loss

        """

        recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="mean")

        return recon_loss

    def loss_function(self, recon_x, x, quantizer_output):

        recon_loss = self._get_vae_loss(recon_x, x)

        vq_loss = quantizer_output.loss

        return (
            recon_loss + vq_loss.mean(dim=0),
            recon_loss,
            vq_loss.mean(dim=0),
        )


@torch.no_grad()
def calculate_perplexity(
    model: CustomVQVAE,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 8192,
) -> float:
    """Compute codebook perplexity over all training patch assignments."""
    model.eval()
    num_embeddings = model.model_config.num_embeddings
    code_counts = torch.zeros(num_embeddings, dtype=torch.float64, device=device)

    def collect_indices(module, inputs, output):
        indices = getattr(output, "min_encoding_indices", None)
        if indices is None:
            indices = getattr(output, "quantized_indices", None)
        if indices is None:
            raise AttributeError(
                "Quantizer output has neither min_encoding_indices nor "
                "quantized_indices; check the installed Pythae version."
            )
        counts = torch.bincount(
            indices.detach().reshape(-1).long(), minlength=num_embeddings
        )
        code_counts.add_(counts.to(device=device, dtype=torch.float64))

    hook = model.quantizer.register_forward_hook(collect_indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    try:
        for batch in loader:
            model({"data": batch["data"].to(device)})
    finally:
        hook.remove()

    total = code_counts.sum()
    if total.item() == 0:
        return float("nan")

    embedding_mean = code_counts / total
    perplexity = torch.exp(
        -torch.sum(embedding_mean * torch.log(embedding_mean + 1e-10))
    )
    return perplexity.item()


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

    - `__getitem__` returns a `DatasetOutput(data=...)` (pythae's own

      `OrderedDict`-with-attribute-access type), not a bare Tensor/ndarray --

      pythae's `TrainingPipeline` requires this once you hand it your own

      `Dataset` subclass (see `pythae/data/datasets.py`'s own `BaseDataset`).

    - Each patch gets an explicit channel dimension

      (`(1, patch_size, patch_size)`), since Conv2d expects `(N, C, H, W)`.

    Returns:

        DatasetOutput: dict-like object with a 'data' key holding the patch

        as a torch.Tensor of shape (1, patch_size, patch_size)

    """

    def __init__(self, patches: np.ndarray, patch_size: int, transform=None):

        self.event_patches_adcs = [
            patch.reshape(patch_size, patch_size) for patch in patches
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
            patch_size=patches_size,
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

    @staticmethod
    def plot_pixel_module(event: PixelDigiEvent, module: PixelModule, patch_size: int = 0, bit_depth: int = 0):
        # hits = event[module]
        # plt.imshow(hits.to_image(), cmap='gist_yarg', origin='lower')
        module_hits = event[module]
        quantized_image = np.full((module.rows, module.cols), np.nan)
        # Fill in the encoding values at the patch coordinates
        quantized_image[module_hits.rows, module_hits.cols] = module_hits.adcs.reshape(-1)

        plt.figure(figsize=(10, 5))
        # plot_pixel_module(event, module)
        im = plt.imshow(quantized_image, cmap='gist_yarg', origin='lower', vmin=0, vmax=255)
        plt.title(f'Event: {event.id_}, DetID: {module.det_id}')
        plt.xlabel('Module column')
        plt.ylabel('Module row')
        clb = plt.colorbar(im, fraction=0.02)
        clb.set_label('ADC value')

        plt.savefig(
            f"pd={patch_size}_bd={bit_depth}_event_pixels_reconstructed.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


@torch.no_grad()
def reconstruct(
    model, data: torch.Tensor, device, batch_size: int = 256
) -> torch.Tensor:

    model.eval()

    outputs = []

    for start in range(0, data.shape[0], batch_size):

        batch = data[start : start + batch_size].to(device)

        out = model({"data": batch})

        recon = torch.sigmoid(out.recon_x)  # logits -> [0, 1] reconstruction

        outputs.append(recon.cpu())

    return torch.cat(outputs, dim=0)


def reconstruct_event(
    model, event: PixelDigiEvent, patch_size: int, device
) -> PixelDigiEvent:
    """Round-trip one event: hits -> patches -> VQ-VAE -> patches -> hits.

    Mirrors the strips script's `reconstruct_event`: patchify the event,
    run every patch through the trained model, round back to ADC counts,
    and rebuild an event from the reconstructed patches so it can be
    compared hit-for-hit against the original with `compare_hit_presence`
    / `compare_matching_hits`.
    """
    blocks = event_hits_to_patches(
        event, patch_size=patch_size, fill_value=0, dtype=np.uint16
    )
    if len(blocks) == 0:
        return event

    # `blocks.as_array()` returns one row per patch (same shape convention
    # `PixelsDataset` already reshapes to `(patch_size, patch_size)`).
    patches = blocks.as_array().reshape(-1, patch_size, patch_size).astype(np.int32)
    x = AddNormalization()(torch.as_tensor(patches))
    recon = reconstruct(model, x.unsqueeze(1), device).squeeze(1).numpy() * ADC_MAX

    # Same decoding as the strips script: round to ADC counts, and a pixel
    # is a hit only if its rounded ADC is > 0.
    adcs = np.round(recon).clip(0, ADC_MAX).astype(np.uint16)
    blocks = replace(blocks, adcs=adcs, occupancy=adcs > 0)
    return event_patches_to_hits(blocks)


def evaluate(
    model,
    events: list[PixelDigiEvent],
    patch_size: int,
    device,
    bit_depth: int,
    debug_module: PixelModule,
) -> dict[str, float]:
    """RMSE over matching hits plus hit-presence precision/recall (pileup_ml metrics)."""
    reconstructed = [reconstruct_event(model, e, patch_size, device) for e in events]

    # plot reconstructed event
    Helper.plot_pixel_module(reconstructed[0], debug_module, patch_size, bit_depth)

    precision, recall = compare_hit_presence(events, reconstructed)
    try:
        rmse = compare_matching_hits(events, reconstructed, metric="rmse")
    except ValueError:  # no pixel is a hit in both original and reconstruction
        rmse = float("nan")
    return {"rmse": rmse, "precision": precision, "recall": recall}


def plot_metric(
    values: dict, bit_depths: list[int], ylabel: str, title: str, out_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for pixel_size, vals in sorted(values.items()):
        ax.plot(bit_depths, vals, marker="o", label=f"Pixel patch size {pixel_size}")
    ax.set_xlabel("Bit depth")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(bit_depths)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_perplexity(values: dict, bit_depths: list[int], out_path: str) -> None:
    """Plot training perplexity by bit depth for each pixel patch size."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for pixel_size, perplexities in sorted(values.items()):
        ax.plot(
            bit_depths,
            perplexities,
            marker="o",
            label=f"Pixel patch size {pixel_size}",
        )
    ax.set_xlabel("Bit depth")
    ax.set_ylabel("Perplexity")
    ax.set_title("VQ-VAE perplexity by pixel patch size")
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
        "/data/cern/pileup_ml/detid_info/detids_bpix.json",
        "/data/cern/pileup_ml/detid_info/detids_fpix.json",
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

    debug_module = pixel_modules[DEBUG_MODULE_INDEX]

    print("INFO: plotting initial test data detector pixel hits")
    Helper.plot_pixel_module(pixels_events_test[0], debug_module)

    data_transform = transforms.Compose([AddNormalization()])

    metric_values = {name: defaultdict(list) for name in ("rmse", "precision", "recall")}
    perplexity_values = defaultdict(list)

    for pixel_size in PIXEL_SIZES:

        pixel_patches_train, pixel_patches_test = Helper.get_patches_sets(
            pixels_events_train, pixels_events_test, pixel_size
        )

        # Rebuild the data for every patch size

        print(f"Preparing train and test data for pixel patch size {pixel_size}")

        vqvae_trainset = PixelsDataset(pixel_patches_train, pixel_size, data_transform)

        for bit_depth in BIT_DEPTHS:

            start_time = datetime.now()

            print(f"Starting VQ-VAE pixels at {start_time:%Y-%m-%d %H:%M:%S}")

            num_embeddings = codebook_size(bit_depth)

            print(
                f"Applying: num_embeddings = {num_embeddings} "
                f"(bit_depth={bit_depth}), pixel_size = {pixel_size}"
            )

            # Fresh seed, config, encoder and decoder for every run so runs are independent

            torch.manual_seed(SEED)

            np.random.seed(SEED)

            model_config = CustomVQVAEConfig(
                input_dim=(CHANNELS, pixel_size, pixel_size),
                latent_dim=32,
                hidden_channels=32,
                num_embeddings=num_embeddings,
                use_ema=True,
                decay=0.9,
                commitment_loss_factor=0.01,
            )

            encoder = CustomEncoderConv(model_config)

            # The decoder needs to know the exact spatial size the encoder's

            # conv stack produced (before its own adaptive pool), so it can

            # pool the quantized feature map back up to that size before its

            # conv-transpose stack runs.

            model_config.encoded_patch_size = encoder.encoded_patch_size

            decoder = CustomDecoderConv(model_config)

            # CustomVQVAE = pythae's VQVAE + BCE-with-logits reconstruction loss

            # (instead of the default MSE).

            model = CustomVQVAE(
                model_config=model_config,
                encoder=encoder,
                decoder=decoder,
            ).to(device)

            print(
                f"embedding_dim resolved by pythae: {model.model_config.embedding_dim}"
            )

            print(f"quantizer: {type(model.quantizer).__name__}")

            training_config = BaseTrainerConfig(
                output_dir=os.path.join(
                    OUTPUT_DIR, f"pixels{pixel_size}_bd{bit_depth}"
                ),
                learning_rate=LEARNING_RATE,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                num_epochs=EPOCHS,
            )

            pipeline = TrainingPipeline(training_config=training_config, model=model)

            pipeline(train_data=vqvae_trainset)

            perplexity = calculate_perplexity(
                model, vqvae_trainset, device, batch_size=BATCH_SIZE
            )
            perplexity_values[pixel_size].append(perplexity)
            print(f"Training perplexity: {perplexity:.4f}")

            # Use the in-memory trained model (avoids reloading from the wrong folder

            # and AutoModel dropping the custom `hidden_channels` config field).

            results = evaluate(
                model, pixels_events_test, pixel_size, device, bit_depth, debug_module
            )
            for name, value in results.items():
                metric_values[name][pixel_size].append(value)

            print(
                f"RMSE (matching hits): {results['rmse']:.2f} ADC | "
                f"precision: {results['precision']:.4f} | recall: {results['recall']:.4f}"
            )

            elapsed = (datetime.now() - start_time).total_seconds()

            print(f"Finished at {datetime.now():%Y-%m-%d %H:%M:%S}")

            print(f"Time difference = {elapsed:.0f} seconds\n------------------")

    # Plot once, after all pixel sizes and bit depths have finished

    print("Plotting metrics")

    plot_metric(metric_values["rmse"], BIT_DEPTHS, "RMSE (ADC)",
                "VQ-VAE RMSE by pixel patch size", PLOT_PATH)
    plot_metric(metric_values["precision"], BIT_DEPTHS, "Precision",
                "VQ-VAE hit precision by pixel patch size", PRECISION_PLOT_PATH)
    plot_metric(metric_values["recall"], BIT_DEPTHS, "Recall",
                "VQ-VAE hit recall by pixel patch size", RECALL_PLOT_PATH)
    plot_perplexity(perplexity_values, BIT_DEPTHS, PERPLEXITY_PLOT_PATH)

    print(
        f"Saved plots to {PLOT_PATH}, {PRECISION_PLOT_PATH}, {RECALL_PLOT_PATH}, "
        f"{PERPLEXITY_PLOT_PATH}"
    )

    print("DONE")


if __name__ == "__main__":

    main()