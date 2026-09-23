"""Plot a REAP expert-saliency report (`.pkl`) as a heatmap.

The report is a `list[list[float]]`: the first dimension is the layer index
and the second is the per-expert saliency value. Each row of the heatmap is a
layer (lowest index at the top), and the columns within a row are that layer's
experts sorted by saliency (highest on the left, lowest on the right).

Usage::

    python plot_reap_report.py report.pkl
    python plot_reap_report.py report.pkl -o saliency.png
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_report(path: Path) -> list[list[float]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def kurtosis(values: np.ndarray) -> float:
    """Excess (Fisher) kurtosis of a 1-D array of values."""
    mean = values.mean()
    std = values.std()
    if std == 0:
        return float("nan")
    return float(np.mean((values - mean) ** 4) / std**4 - 3)


def plot_heatmap(
    report: list[list[float]],
    title: str,
    out_path: Path,
    layerwise_norm: bool = False,
) -> None:
    # Sort each layer's experts by saliency, highest (left) to lowest (right).
    matrix = np.array([sorted(row, reverse=True) for row in report], dtype=float)

    # Kurtosis is computed on the raw per-layer values (before normalization);
    # it is scale-invariant, so normalizing would not change it.
    kurtoses = [kurtosis(row) for row in matrix]

    if layerwise_norm:
        # Divide each layer's values by that layer's maximum value.
        row_max = matrix.max(axis=1, keepdims=True)
        matrix = np.divide(
            matrix, row_max, out=np.zeros_like(matrix), where=row_max != 0
        )

    fig, ax = plt.subplots(figsize=(10, 0.4 * len(report) + 2))
    cax = ax.imshow(matrix, aspect="auto", cmap="inferno", interpolation="nearest")

    if layerwise_norm:
        ax.set_title(title + "\nnormalized per layer (value / layer max)")
    else:
        ax.set_title(title)
    ax.set_xlabel("Expert rank by saliency (high -> low)")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(range(1, matrix.shape[1] + 1))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(range(matrix.shape[0]))

    if layerwise_norm:
        # Label each layer (row) with its kurtosis on the right-hand side.
        for i, k in enumerate(kurtoses):
            ax.text(
                matrix.shape[1] - 0.3,
                i,
                f"k={k:.2f}",
                va="center",
                ha="left",
                fontsize=7,
                clip_on=False,
            )

    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved heatmap to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a REAP saliency report as a heatmap"
    )
    parser.add_argument("report", type=Path, help="Path to the .pkl REAP report")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <report>.png)",
    )
    parser.add_argument(
        "--layerwise-norm",
        action="store_true",
        help="Divide each saliency value by the highest value in its layer",
    )
    args = parser.parse_args()

    report = load_report(args.report)
    out_path = args.output or args.report.with_suffix(".png")
    title = f"{args.report.stem} REAP Saliency"
    plot_heatmap(report, title, out_path, layerwise_norm=args.layerwise_norm)


if __name__ == "__main__":
    main()
