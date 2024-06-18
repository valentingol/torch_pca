"""Test the n_components parameter of PCA."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
import pytest
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


def test_float() -> None:
    """Test when n_components is float."""
    inputs = torch.load("tests/input_data2.pt").to(torch.float32)
    torch_model = PCA(n_components=0.8)
    sklearn_model = PCA(n_components=0.8)
    torch_model.fit(inputs)
    sklearn_model.fit(inputs)
    check.equal(torch_model.n_components_, sklearn_model.n_components_)
    torch_model = PCA(n_components=0.9)
    sklearn_model = PCA(n_components=0.9)
    torch_model.fit(inputs.T + 2.0)
    sklearn_model.fit(inputs.T + 2.0)
    check.equal(torch_model.n_components_, sklearn_model.n_components_)


def test_wrong_cases() -> None:
    """Test when n_components is not valid."""
    inputs = torch.load("tests/input_data2.pt").to(torch.float32)
    with pytest.raises(
        ValueError,
        match=".*1.1.*",
    ):
        PCA(n_components=1.1).fit(inputs)


def test_mle() -> None:
    """Test when n_components is set to 'mle'."""
    inputs = torch.load("tests/input_data3.pt").to(torch.float32)
    torch_model = PCA(n_components="mle").fit(inputs)
    sklearn_model = PCA(n_components="mle").fit(inputs)
    check.equal(torch_model.n_components_, sklearn_model.n_components_)
