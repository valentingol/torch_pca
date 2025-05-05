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
        t_0 = time()
        PCA(n_components=50).fit_transform(inputs)
        torch_times.append(round(time() - t_0, 4))
        t_0 = time()
        PCA_sklearn(n_components=50).fit_transform(inputs)
        sklearn_times.append(round(time() - t_0, 4))
    ticks = np.arange(len(configs))
    labels = [f"n_samples={config[0]}, n_features={config[1]}" for config in configs]
    width = 0.35
    fig, axis = plt.subplots()
    rects1 = axis.bar(ticks - width / 2, torch_times, width, label="Pytorch PCA")
    rects2 = axis.bar(ticks + width / 2, sklearn_times, width, label="Sklearn PCA")
    axis.set_ylabel("Time of execution (s)")
    axis.set_title("Comparison of execution time between Pytorch and Sklearn PCA.")
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)
    axis.legend()
    autolabel(rects1, axis)
    autolabel(rects2, axis)
    fig.tight_layout()
    plt.show()


def autolabel(rects: list, axis: plt.Axes) -> None:
    """Attach a text label above each bar in *rects*, displaying its height.

    From https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    """
    for rect in rects:
        height = rect.get_height()
        axis.annotate(
            str(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


if __name__ == "__main__":
    main()
