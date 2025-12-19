from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


matplotlib.rcParams.update(
    {
        "figure.figsize": (7.0, 4.5),
        "axes.grid": True,
        "grid.alpha": 0.2,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    }
)


def set_style() -> None:
    sns.set_theme(style="whitegrid")


def save_figure(path: str | Path, caption: Optional[str] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    if caption:
        caption_path = path.with_suffix("")
        caption_path = caption_path.parent / (caption_path.name + "_caption.txt")
        caption_path.write_text(caption + "\n", encoding="utf-8")
    plt.close()


def heatmap(data, title: str, xlabel: str, ylabel: str, cbar_label: str) -> None:
    ax = sns.heatmap(data, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.set_label(cbar_label)


def lineplot(x, ys, labels, title: str, xlabel: str, ylabel: str) -> None:
    for y, label in zip(ys, labels, strict=True):
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
