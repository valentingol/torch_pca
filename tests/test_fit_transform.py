"""Test results of the PCA class (fit and transform)."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
import pytest
import pytest_check as check
import torch
from sklearn.decomposition import PCA as PCA_sklearn

from torch_pca import PCA


def test_fullsvd() -> None:
    """Test basic full SVD."""
    input_1 = torch.load("tests/input_data.pt").to(torch.float32) + 2.0
    torch_model = PCA(
        n_components=2,
        svd_solver="full",
    ).fit(input_1)
    sklearn_model = PCA_sklearn(
        n_components=2,
        svd_solver="full",
        whiten=False,
    ).fit(input_1)
    check.is_true(
        torch.allclose(
            torch.tensor(sklearn_model.components_, dtype=torch.float32),
            torch_model.components_,
            rtol=1e-5,
            atol=1e-5,
        )
    )
    torch_outs = torch_model.transform(input_1)
    sklearn_outs = torch.tensor(sklearn_model.transform(input_1), dtype=torch.float32)
    check.is_true(torch.allclose(torch_outs, sklearn_outs, rtol=1e-5, atol=1e-5))

    check.is_true(
        torch.allclose(
            torch_model.fit_transform(input_1),
            torch_outs,
            rtol=1e-5,
            atol=1e-5,
        )
    )
    # New data
    input_2 = torch.load("tests/input_data2.pt").to(torch.float32) - 1.0
    torch_outs = torch_model.transform(input_2)
    sklearn_outs = torch.tensor(sklearn_model.transform(input_2), dtype=torch.float32)
    check.is_true(torch.allclose(torch_outs, sklearn_outs, rtol=1e-5, atol=1e-5))

    # Fail if not fitted
    model = PCA(n_components=2)
    with pytest.raises(ValueError, match="PCA not fitted when calling transform..*"):
        model.transform(input_1)


def test_centering() -> None:
    """Test centering options of transform."""
    inputs = torch.load("tests/input_data.pt").to(torch.float32)
    centered_inputs = inputs - inputs.mean(dim=0, keepdim=True)
    inputs_1 = centered_inputs + 1.0
    model = PCA(n_components=2).fit(inputs_1)
    inputs_2 = centered_inputs - 1.0
    check.is_true(
        torch.allclose(
            model.transform(inputs_1),
            model.transform(inputs_2, center="input"),
            rtol=1e-5,
            atol=1e-5,
        )
    )
    check.is_true(
        torch.allclose(
            model.transform(inputs_1),
            model.transform(centered_inputs, center="none"),
            rtol=1e-5,
            atol=1e-5,
        )
    )
    # Unkown centering
    with pytest.raises(ValueError, match="Unknown centering.*"):
        model.transform(inputs_1, center="UNKNOWN")


def test_covariance_eigh() -> None:
    """Test SVD based on covariance matrix."""
    input_1 = torch.load("tests/input_data.pt").to(torch.float32) + 2.0
    torch_model = PCA(n_components=2, svd_solver="covariance_eigh").fit(input_1)
    sklearn_model = PCA_sklearn(
        n_components=2,
        svd_solver="covariance_eigh",
        whiten=False,
    ).fit(input_1)
    for attr_name in ["components_", "explained_variance_", "singular_values_"]:
        attr_torch = getattr(torch_model, attr_name)
        attr_sklearn = getattr(sklearn_model, attr_name)
        check.is_true(
            torch.allclose(
                attr_torch,
                torch.tensor(attr_sklearn, dtype=torch.float32),
                rtol=1e-5,
                atol=1e-5,
            ),
            f"Failed for attribute {attr_name}",
        )


def test_fail_svd() -> None:
    """Test unknown SVD solver."""
    input_1 = torch.randn(200, 10)
    with pytest.raises(ValueError, match="Unknown SVD solver.*"):
        PCA(n_components=2, svd_solver="UNKNOWN").fit(input_1)


def test_auto_svd() -> None:
    """Test auto SVD solver."""
    input_1 = torch.randn(200, 10)
    model = PCA(n_components=2).fit(input_1)
    check.equal(model.svd_solver_, "covariance_eigh")
    input_2 = torch.randn(50, 8)
    model = PCA(n_components=2).fit(input_2)
    check.equal(model.svd_solver_, "full")
    input_3 = torch.randn(10, 1200)
    model = PCA(n_components=2).fit(input_3)
    check.equal(model.svd_solver_, "full")
