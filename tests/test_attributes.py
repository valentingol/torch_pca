"""Test attributes of the PCA object."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
import numpy as np
import pytest_check as check
import torch
from sklearn.decomposition import PCA as PCA_sklearn

from torch_pca import PCA


def test_attr() -> None:
    """Check the attributes match sklearn's PCA attributes."""
    inputs = torch.load("tests/input_data.pt").to(torch.float32)
    torch_model = PCA(n_components=2).fit(inputs)
    sklearn_model = PCA_sklearn(
        n_components=2,
        svd_solver="full",
        whiten=False,
    ).fit(inputs)
    for attr_name in [
        "mean_",
        "n_components_",
        "components_",
        "explained_variance_",
        "explained_variance_ratio_",
        "singular_values_",
        "n_samples_",
        "noise_variance_",
        "n_features_in_",
    ]:
        attr = getattr(torch_model, attr_name)
        attr_sklearn = getattr(sklearn_model, attr_name)
        if attr_name == "explained_variance_":
            print(attr, attr_sklearn)
        if isinstance(attr, torch.Tensor):
            check.is_true(
                torch.allclose(
                    attr,
                    torch.tensor(attr_sklearn, dtype=torch.float32),
                    rtol=1e-5,
                    atol=1e-5,
                ),
                f"Attribute {attr_name} does not match.",
            )
        else:
            check.is_true(
                np.isclose(
                    attr,
                    attr_sklearn,
                    rtol=1e-5,
                    atol=1e-5,
                ),
                f"Attribute {attr_name} does not match.",
            )
