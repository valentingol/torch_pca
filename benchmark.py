"""Comparison between sklearn and torch PCA models."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.

from time import time

# NOTE: requires matplotlib (not in requirements(-dev).txt)
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA as PCA_sklearn

from torch_pca import PCA


def main() -> None:
    """Measure and compare the time of execution of the PCA."""
    configs = [(75, 75), (100, 2000), (10_000, 500)]
    torch_times, sklearn_times = [], []
    for config in configs:
        inputs = torch.randn(*config)
        t0 = time()
        PCA(n_components=50).fit_transform(inputs)
        torch_times.append(round(time() - t0, 4))
        t0 = time()
        PCA_sklearn(n_components=50).fit_transform(inputs)
        sklearn_times.append(round(time() - t0, 4))
    ticks = np.arange(len(configs))
    labels = [f"n_samples={config[0]}, n_features={config[1]}" for config in configs]
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(ticks - width / 2, torch_times, width, label="Pytorch PCA")
    rects2 = ax.bar(ticks + width / 2, sklearn_times, width, label="Sklearn PCA")
    ax.set_ylabel("Time of execution (s)")
    ax.set_title("Comparison of execution time between Pytorch and Sklearn PCA.")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.legend()
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    fig.tight_layout()
    plt.show()


def autolabel(rects: list, ax: plt.Axes) -> None:
    """Attach a text label above each bar in *rects*, displaying its height.

    From https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            str(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


if __name__ == "__main__":
    main()
