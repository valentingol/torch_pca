"""Test PCA on device/dtype."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
import pytest
import pytest_check as check
import torch

from torch_pca import PCA


def test_to_dtype_device() -> None:
    """Test put model on cpu and float16."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_1 = torch.load("tests/input_data.pt").to(torch.float32).to(device)
    torch_model = PCA(
        n_components=2,
        svd_solver="full",
    ).fit(input_1)
    for attr in dir(torch_model):
        if not attr.startswith("__") and isinstance(
            getattr(torch_model, attr), torch.Tensor
        ):
            check.is_true(
                getattr(torch_model, attr).dtype == torch.float32,
            )
    torch_model.to(torch.float16)
    for attr in dir(torch_model):
        if not attr.startswith("__") and isinstance(
            getattr(torch_model, attr), torch.Tensor
        ):
            check.is_true(
                getattr(torch_model, attr).dtype == torch.float16,
            )
    torch_model.to("cpu")
    torch_model.transform(input_1.to("cpu").to(torch.float16))

    with pytest.raises(
        ValueError,
        match="PCA not fitted when calling `.to.*",
    ):
        PCA(n_components=2).to("cpu")
    with pytest.raises(
        ValueError,
        match="Unknown argument type in `args`.*",
    ):
        # Complex number -> invalid type
        torch_model.to(1j)
    with pytest.raises(
        RuntimeError,
        match="Expected one of cpu, cuda.*",
    ):
        torch_model.to("unknown")
