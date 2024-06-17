"""PyTorch PCA.

Principal Component Anlaysis (PCA) in PyTorch. The intention is to provide a
simple and easy to use implementation of PCA in PyTorch, the most similar to
the `sklearn`'s PCA as possible (in terms of API and, of course, output).
"""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
from torch_pca._version import version, version_tuple
from torch_pca.pca_main import PCA

__all__ = ["PCA", "version", "version_tuple"]
