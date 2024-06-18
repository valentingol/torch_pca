"""Test inverse transform."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
import pytest
import pytest_check as check
import torch

from torch_pca import PCA


def test_inv_transform() -> None:
    """Test inverse transform."""
    inputs = torch.load("tests/input_data.pt").to(torch.float32)
    model = PCA(n_components=2)
    transformed = model.fit_transform(inputs)
    de_transformed = model.inverse_transform(transformed)
    check.is_true(
        torch.allclose(
            inputs,
            de_transformed,
            rtol=1e-5,
            atol=1e-5,
        )
    )
    with pytest.raises(
        ValueError, match="PCA not fitted when calling inverse_transform..*"
    ):
        PCA(n_components=2).inverse_transform(inputs)
