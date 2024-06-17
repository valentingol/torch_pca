"""Main module for PCA."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
from typing import Optional

import torch
from torch import Tensor

from torch_pca.svd import NComponentsType, choose_svd_solver, svd_flip


class PCA:
    """Principal Component Analysis (PCA).

    Works with PyTorch tensors.
    API similar to sklearn.decomposition.PCA.

    Parameters
    ----------
    n_components: int | float | str | None, optional
        Number of components to keep.

        * If int, number of components to keep.
        * If float (should be between 0.0 and 1.0), the number of components
          to keep is determined by the cumulative percentage of variance
          explained by the components until the proportion is reached.
        * If "mle", the number of components is selected using Minka's MLE.
        * If None, all components are kept: n_components = min(n_samples, n_features).
        By default, n_components=None.

    svd_solver: str, optional
        One of {'auto', 'full', 'covariance_eigh'}

        * 'auto': the solver is selected automatically based on the shape of the input.
        * 'full': Run exact full SVD with torch.linalg.svd
        * 'covariance_eigh': Compute the covariance matrix and take
          the eigenvalues decomposition with torch.linalg.eigh.
          Most efficient for small n_features and large n_samples.
        By default, svd_solver='auto'.
    """

    def __init__(
        self,
        n_components: NComponentsType = None,
        svd_solver: str = "auto",
    ):
        self.components_: Optional[Tensor] = None
        """Principal axes in feature space."""
        self.explained_variance_: Optional[Tensor] = None
        """The amount of variance explained by each of the selected components."""
        self.explained_variance_ratio_: Optional[Tensor] = None
        """Percentage of variance explained by each of the selected components."""
        self.mean_: Optional[Tensor] = None
        """Mean of the input data during fit."""
        self.n_components_ = n_components
        """Number of components to keep."""
        self.n_features_in_: int = -1
        """Number of features in the input data."""
        self.n_samples_: int = -1
        """Number of samples seen during fit."""
        self.noise_variance_: Optional[Tensor] = None
        """The estimated noise covariance."""
        self.singular_values_: Optional[Tensor] = None
        """The singular values corresponding to each of the selected components."""
        self.svd_solver_ = svd_solver
        """Solver to use for the PCA computation."""

        if self.svd_solver_ not in ["auto", "full", "covariance_eigh"]:
            raise ValueError(
                "Unknown SVD solver. `svd_solver` should be one of "
                "'auto', 'full', 'covariance_eigh'."
            )

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
        if self.svd_solver_ == "auto":
            self.svd_solver_ = choose_svd_solver(inputs, self.n_components_)
        self.mean_ = inputs.mean(dim=-2, keepdim=True)
        self.n_samples_, self.n_features_in_ = inputs.shape[-2:]
        if self.svd_solver_ == "full":
            inputs_centered = inputs - self.mean_
            u_mat, coefs, vh_mat = torch.linalg.svd(  # pylint: disable=E1102
                inputs_centered,
                full_matrices=False,
            )
            explained_variance_ = coefs**2 / (inputs.shape[-2] - 1)
        elif self.svd_solver_ == "covariance_eigh":
            covariance = inputs.T @ inputs
            delta = self.n_samples_ * torch.transpose(self.mean_, -2, -1) * self.mean_
            covariance -= delta
            covariance /= self.n_samples_ - 1
            eigenvals, eigenvecs = torch.linalg.eigh(covariance)
            # Fix eventual numerical errors
            eigenvals[eigenvals < 0.0] = 0.0
            # Inverted indices
            idx = range(eigenvals.size(0) - 1, -1, -1)
            idx = torch.LongTensor(idx)
            explained_variance_ = eigenvals.index_select(0, idx)
            # Compute equivalent variables to full SVD output
            vh_mat = eigenvecs.T.index_select(0, idx)
            coefs = torch.sqrt(explained_variance_ * (self.n_samples_ - 1))
            u_mat = None
        _, vh_mat = svd_flip(u_mat, vh_mat)
        total_var = torch.sum(explained_variance_)
        explained_variance_ratio_ = explained_variance_ / total_var
        if self.n_components_ is None:
            self.n_components_ = min(inputs.shape[-2:])
        if not isinstance(self.n_components_, int):
            raise ValueError("`n_components` value not supported.")
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
