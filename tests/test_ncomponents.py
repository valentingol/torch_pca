"""Test the n_components parameter of PCA."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
import pytest_check as check
import torch

from torch_pca import PCA


def test_int_none() -> None:
    """Test when n_components is set to an int or None."""
    inputs = torch.load("tests/input_data.pt").to(torch.float32)  # shape (3, 10)
    model = PCA(n_components=2)
    check.equal(model.n_components_, 2)
    model = PCA(n_components=None)
    check.equal(model.n_components_, None)
    model.fit(inputs)
    check.equal(model.n_components_, 3)
    model = PCA().fit(inputs)
    check.equal(model.n_components_, 3)
