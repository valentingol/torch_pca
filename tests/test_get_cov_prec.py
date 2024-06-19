"""Tests around covariance, precision and score methods."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
import pytest_check as check
import torch
from sklearn.decomposition import PCA as PCA_sklearn

from torch_pca import PCA


def test_get_covariance() -> None:
    """Test get_covariance method."""
    inputs = torch.load("tests/input_data.pt").to(torch.float32)
    torch_model = PCA(n_components=2).fit(inputs)
    sklearn_model = PCA_sklearn(n_components=2).fit(inputs)
    check.is_true(
        torch.allclose(
            torch.tensor(sklearn_model.get_covariance(), dtype=torch.float32),
            torch_model.get_covariance(),
            rtol=1e-5,
            atol=1e-5,
        )
    )


def test_precision() -> None:
    """Test get_precision method."""
    inputs = torch.load("tests/input_data3.pt").to(torch.float32)
    torch_model = PCA(n_components=2).fit(inputs)
    sklearn_model = PCA_sklearn(n_components=2).fit(inputs)
    print(
        torch_model.get_precision(),
    )
    print(
        torch.tensor(sklearn_model.get_precision(), dtype=torch.float32),
    )
    check.is_true(
        torch.allclose(
            torch.tensor(sklearn_model.get_precision(), dtype=torch.float32),
            torch_model.get_precision(),
            rtol=1e-5,
            atol=1e-5,
        )
    )


def test_scores() -> None:
    """Test score-related methods."""
    inputs = torch.load("tests/input_data3.pt").to(torch.float32)
    for whiten in [True, False]:
        torch_model = PCA(n_components=2, whiten=whiten).fit(inputs)
        sklearn_model = PCA_sklearn(n_components=2, whiten=whiten).fit(inputs)
        check.is_true(
            torch.allclose(
                torch.tensor(sklearn_model.score_samples(inputs), dtype=torch.float32),
                torch_model.score_samples(inputs),
                rtol=1e-5,
                atol=1e-5,
            ),
            f"Wrong with whiten={whiten}",
        )
        check.is_true(
            torch.allclose(
                torch.tensor(sklearn_model.score(inputs), dtype=torch.float32),
                torch_model.score(inputs),
                rtol=1e-5,
                atol=1e-5,
            ),
            f"Wrong with whiten={whiten}",
        )
