"""Test PCA with GPU and different dtypes."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
import pytest_check as check
import torch

from torch_pca import PCA


def test_gpu() -> None:
    """Test with GPU and different dtypes."""
    inputs = torch.load("tests/input_data.pt").to("cuda:0")
    for dtype in [torch.float32, torch.float16, torch.float64]:
        inputs = inputs.to(dtype)
        out1 = PCA(svd_solver="full").fit_transform(inputs)
        out2 = PCA(svd_solver="covariance_eigh").fit_transform(inputs)
        out3 = PCA(svd_solver="randomized", random_state=0).fit_transform(inputs)
        for out in [out1, out2, out3]:
            check.equal(str(out.device), "cuda:0")
            check.equal(out.dtype, dtype)
