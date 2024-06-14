"""Main module for PCA."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
from typing import Optional, Union

import torch
from torch import Tensor

from torch_pca.svd import svd_flip

N_COMPONENTS_TYPE = Union[int, float, None, str]


class PCA:
    """Principal Component Analysis (PCA).

    Works with PyTorch tensors.
    API similar to sklearn.decomposition.PCA.

    Parameters
    ----------
    n_components: int | float | str | None
        Number of components to keep.
        - If int, number of components to keep.
        - If float (should be between 0.0 and 1.0), the number of components
          to keep is determined by the cumulative percentage of variance
          explained by the components until the proportion is reached.
        - If "mle", the number of components is selected using Minka's MLE.
        - If None, all components are kept: n_components = min(n_samples, n_features).
        By default, n_components=None.
    """

    def __init__(self, n_components: N_COMPONENTS_TYPE = None):
        self.mean_: Optional[Tensor] = None
        self.n_components_ = n_components
        self.components_: Optional[Tensor] = None
        self.explained_variance_: Optional[Tensor] = None
        self.explained_variance_ratio_: Optional[Tensor] = None
        self.singular_values_: Optional[Tensor] = None
        self.n_samples_: Optional[int] = None
        self.noise_variance_: Optional[Tensor] = None
        self.n_features_in_: Optional[int] = None

    def fit_transform(self, inputs: Tensor) -> Tensor:
        """Fit the PCA model and apply the dimensionality reduction.

        Parameters
        ----------
        inputs : Tensor
            Input data of shape (n_samples, n_features).

        Returns
        -------
        transformed : Tensor
            Transformed data.
        """
        self.fit(inputs)
        transformed = self.transform(inputs)
        return transformed

    def fit(self, inputs: Tensor) -> "PCA":
        """Fit the PCA model and return it.

        Parameters
        ----------
        inputs : Tensor
            Input data of shape (n_samples, n_features).
        """
        self.n_samples_, self.n_features_in_ = inputs.shape[-2:]
        self.mean_ = inputs.mean(dim=-2, keepdim=True)
        inputs_centered = inputs - self.mean_
        u_mat, coefs, vh_mat = torch.linalg.svd(  # pylint: disable=E1102
            inputs_centered,
            full_matrices=False,
        )
        u_mat, vh_mat = svd_flip(u_mat, vh_mat)

        explained_variance_ = coefs**2 / (inputs.shape[-2] - 1)
        total_var = torch.sum(explained_variance_)
        explained_variance_ratio_ = explained_variance_ / total_var
        if self.n_components_ is None:
            self.n_components_ = min(inputs.shape[-2:])
        assert isinstance(
            self.n_components_, int
        ), "Internal error with n_components_ value, please report."
        self.components_ = vh_mat[: self.n_components_]
        self.explained_variance_ = explained_variance_[: self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio_[: self.n_components_]
        self.singular_values_ = coefs[: self.n_components_]
        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        self.noise_variance_ = (
            torch.mean(explained_variance_[self.n_components_ :])
            if self.n_components_ < min(inputs.shape[-2:])
            else torch.tensor(0.0)
        )
        return self

    def transform(self, inputs: Tensor, center: str = "fit") -> Tensor:
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        inputs : Tensor
            Input data of shape (n_samples, n_features).
        center : str
            One of 'fit', 'input' or 'none'.
            Precise how to center the data.
            - 'fit': center the data using the mean fitted during `fit` (default).
            - 'input': center the data using the mean of the input data.
            - 'none': do not center the data.
            By default, 'fit' (as sklearn PCA implementation)

        Returns
        -------
        transformed : Tensor
            Transformed data.
        """
        if self.components_ is None:
            raise ValueError(
                "PCA not fitted when calling transform. "
                "Please call `fit` or `fit_transform` first."
            )
        transformed = inputs @ self.components_.T
        if center == "fit":
            transformed -= self.mean_ @ self.components_.T
        elif center == "input":
            transformed -= inputs.mean(dim=-2, keepdim=True) @ self.components_.T
        elif center != "none":
            raise ValueError(
                "Unknown centering, `center` argument should be "
                "one of 'fit', 'input' or 'none'."
            )
        return transformed
