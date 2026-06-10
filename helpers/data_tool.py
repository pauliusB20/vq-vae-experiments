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
            d, 
            vq_n_embed
        ) = encoded_size
        
        # TODO: skaiciuoti bit'us
        vq_bits = d * np.log2(vq_n_embed)
        
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
    
class CMSPlots:
    X_MAX = 2
    
    def __init__(self) -> None:
        pass
    
    def plot_patch(self, patch: np.array, title: str) -> None:
        plt.imshow(patch, cmap='gist_yarg', origin='lower')
        plt.title(title)
        plt.show()
        
    def plot_curve(
        self,
        y_values: list[float], 
        x_values: int, 
        y_title: str, 
        title: str,
        x_label: str,
        marker: str=None
    ) -> None:
        print(title + " values over epochs")
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(8, 5))
        plt.plot(x_values, y_values, marker=marker)
        plt.xlabel(x_label)
        plt.ylabel(y_title)
        plt.legend()
        plt.title(title)
        plt.show()
        
    def plot_curves(self, values_to_plot: list[tuple]) -> None:
        size = len(values_to_plot)
        y_max = (size // 2) + (size % 2)

        fig, axes = plt.subplots(y_max, self.X_MAX, figsize=(12, 8))
        i = 0

        for y in range(y_max):
            for x in range(self.X_MAX):
                if i < len(values_to_plot):
                    (
                        epochs, 
                        y_values, 
                        title, 
                        y_label
                    ) = values_to_plot[i]
                    
                    ax = axes.flat[i]
                    ax.plot(
                        epochs,
                        y_values,
                    )
                    ax.grid(True)
                    ax.set_title(title)
                    ax.set_xlabel("x")
                    ax.set_ylabel(f"y, {y_label}")
                    
                    for spine in ax.spines.values():
                        spine.set_linewidth(2)
                        
                    i += 1

        for ax in axes.flat[len(values_to_plot):]:
            ax.axis("off")
        
        plt.tight_layout()
        plt.show()
        
    def plot_patches(self, patch_rows: list[tuple], title: str) -> None:
        cols = len(patch_rows[0][0])
        rows = len(patch_rows)

        fig = plt.figure(figsize=(2.2 * cols, 8.5))

        gs = GridSpec(
            rows,
            cols,
            figure=fig,
            height_ratios=[1.2, 1.5, 1.2],
            wspace=0.25,
            hspace=0.85
        )

        fig.suptitle(
            title,
            fontsize=16,
            fontweight="bold",
            y=0.98
        )
        
        global_abs_max = max(
            p.detach().cpu().abs().max().item()
            if hasattr(p, "detach") else abs(p).max()
            for p in patch_rows[1][0]
        )

        for i in range(rows):
            patch_row, patch_row_title = patch_rows[i]

            # Row title aligned above each row
            row_top = gs[i, :].get_position(fig).y1
            fig.text(
                0.5,
                row_top + 0.035,
                patch_row_title,
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold"
            )

            for j in range(cols):
                ax = fig.add_subplot(gs[i, j])

                patch = patch_row[j]

                # Keep your current row-1 transpose logic
                if i == 1:
                    patch = patch.T

                    im = ax.imshow(
                        patch,
                        cmap="coolwarm",
                        origin="lower",
                        aspect="auto",
                        interpolation="nearest",
                        vmin=-global_abs_max,
                        vmax=global_abs_max
                    )

                    ax.set_box_aspect(1.5)


                else:
                    im = ax.imshow(
                        patch,
                        cmap="gist_yarg",
                        origin="lower",
                        aspect="equal"
                    )

                    # Square input/output patches
                    ax.set_box_aspect(1.0)

                ax.set_xticks([])
                ax.set_yticks([])

                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.5)

        plt.show()

        
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
        
    
    
    
    