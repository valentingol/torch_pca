"""Main module for PCA."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
from typing import Any, Optional

import torch
from torch import Tensor
from torch._prims_common import DeviceLikeType

from torch_pca.ncompo import NComponentsType, find_ncomponents
from torch_pca.svd import choose_svd_solver, svd_flip


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
        #: Principal axes in feature space.
        self.components_: Optional[Tensor] = None
        #: The amount of variance explained by each of the selected components.
        self.explained_variance_: Optional[Tensor] = None
        #: Percentage of variance explained by each of the selected components.
        self.explained_variance_ratio_: Optional[Tensor] = None
        #: Mean of the input data during fit.
        self.mean_: Optional[Tensor] = None
        #: Number of components to keep.
        self.n_components_: NComponentsType = n_components
        #: Number of features in the input data.
        self.n_features_in_: int = -1
        #: Number of samples seen during fit.
        self.n_samples_: int = -1
        #: The estimated noise covariance.
        self.noise_variance_: Optional[Tensor] = None
        #: Singular values corresponding to each of the selected components.
        self.singular_values_: Optional[Tensor] = None
        #: Solver to use for the PCA computation.
        self.svd_solver_: str = svd_solver

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
            self.svd_solver_ = choose_svd_solver(
                inputs=inputs,
                n_components=self.n_components_,
            )
        self.mean_ = inputs.mean(dim=-2, keepdim=True)
        self.n_samples_, self.n_features_in_ = inputs.shape[-2:]
        if self.svd_solver_ == "full":
            inputs_centered = inputs - self.mean_
            u_mat, coefs, vh_mat = torch.linalg.svd(  # pylint: disable=E1102
                inputs_centered,
                full_matrices=False,
            )
            explained_variance = coefs**2 / (inputs.shape[-2] - 1)
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
            explained_variance = eigenvals.index_select(0, idx)
            # Compute equivalent variables to full SVD output
            vh_mat = eigenvecs.T.index_select(0, idx)
            coefs = torch.sqrt(explained_variance * (self.n_samples_ - 1))
            u_mat = None
        _, vh_mat = svd_flip(u_mat, vh_mat)  # pylint: disable=E0601
        total_var = torch.sum(explained_variance)
        explained_variance_ratio = explained_variance / total_var
        self.n_components_ = find_ncomponents(
            n_components=self.n_components_,
            inputs=inputs,
            n_samples=self.n_samples_,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
        )
        self.components_ = vh_mat[: self.n_components_]
        self.explained_variance_ = explained_variance[: self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components_]
        self.singular_values_ = coefs[: self.n_components_]
        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        self.noise_variance_ = (
            torch.mean(explained_variance[self.n_components_ :])
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

            * 'fit': center the data using the mean fitted during `fit` (default).
            * 'input': center the data using the mean of the input data.
            * 'none': do not center the data.

            By default, 'fit' (as sklearn PCA implementation)

        Returns
        -------
        transformed : Tensor
            Transformed data of shape (n_samples, n_components).
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

    def inverse_transform(self, inputs: Tensor) -> Tensor:
        """De-transform transformed data.

        Parameters
        ----------
        inputs : Tensor
            Transformed data of shape (n_samples, n_components).

        Returns
        -------
        de_transformed : Tensor
            De-transformed data of shape (n_samples, n_features)
            where n_features is the number of features in the input data
            before applying transform.
        """
        if self.components_ is None:
            raise ValueError(
                "PCA not fitted when calling inverse_transform. "
                "Please call `fit` or `fit_transform` first."
            )
        de_transformed = inputs @ self.components_ + self.mean_
        return de_transformed

    def to(self, *args: Any, **kwargs: Any) -> None:
        """Move the model to the specified device/dtype.

        Call the native PyTorch `.to()` method on all tensors, parameters
        and NN modules to move the model to the specified device and/or dtype.

        Parameters
        ----------
        args : Any
            Positional arguments to pass to the `.to()` method.
        kwargs : Any
            Keyword arguments to pass to the `.to()` method.
            They can be:
            device : torch.DeviceLikeType
                Device to move the model to.
            dtype : torch.dtype
                Data type to move the model to.
            non_blocking : bool, optional
                If True, the operation will be non-blocking.
                By default, False.
            copy : bool, optional
            memory_format : torch.memory_format, optional

        Note
        ----
            By default, the parameters dtype and device are the same as
            the input data dtype and device during the fit.
            This method is used if want you to change the dtype and/or device
            of the model after the fit. For instance if you fit the model
            on GPU and want to make inference on CPU.

        Warning
        -------
            Require the model to be fitted first.
        """
        to_args = {}
        for arg in args:
            if isinstance(arg, torch.dtype):
                to_args["dtype"] = arg
            elif isinstance(arg, DeviceLikeType):
                to_args["device"] = arg
            else:
                raise ValueError(
                    "Unknown argument type in `args`, "
                    "should be one of `torch.dtype` or `torch.DeviceLikeType`."
                )
        to_args.update(kwargs)
        self._to(**to_args)

    def _to(
        self,
        device: Optional[DeviceLikeType] = None,
        dtype: Optional[torch.dtype] = None,
        *,
        non_blocking: bool = False,
        **kwargs: dict,
    ) -> None:
        """Move the model to the specified device/dtype.

        Call the native PyTorch `.to()` method on all tensors, parameters
        and NN modules to move the model to the specified device and/or dtype.
        """
        if self.components_ is None:
            raise ValueError(
                "PCA not fitted when calling `.to()`. "
                "Please call `fit` or `fit_transform` first."
            )
        attr_list = list(
            filter(
                lambda x: not x.startswith("__"),
                dir(self),
            )
        )
        for attr_name in attr_list:
            attr_value = getattr(self, attr_name)
            if isinstance(
                attr_value, (torch.Tensor, torch.nn.Parameter, torch.nn.Module)
            ):
                setattr(
                    self,
                    attr_name,
                    attr_value.to(
                        device=device, dtype=dtype, non_blocking=non_blocking, **kwargs
                    ),
                )
