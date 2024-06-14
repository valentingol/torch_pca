"""Functions related to SVD."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
from typing import Tuple

import torch
from torch import Tensor


def svd_flip(u_mat: Tensor, vh_mat: Tensor) -> Tuple[Tensor, Tensor]:
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that
    the loadings in the rows in V^H that are largest in absolute
    value are always positive.

    Parameters
    ----------
    u_mat : ndarray
        U matrix in the SVD output (U * diag(S) * V^H)

    vh_mat : ndarray
        V^H matrix in the SVD output (U * diag(S) * V^H)

    Returns
    -------
    u_mat : ndarray
        Adjusted U v.

    vh_mat : ndarray
         Adjusted V^H matrix.
    """
    max_abs_v_rows = torch.argmax(torch.abs(vh_mat), dim=1)
    shift = torch.arange(vh_mat.shape[0])
    indices = max_abs_v_rows + shift * vh_mat.shape[1]
    flat_vh = torch.reshape(vh_mat, (-1,))
    signs = torch.sign(torch.take_along_dim(flat_vh, indices, dim=0))
    if u_mat is not None:
        u_mat *= signs[None, :]
    vh_mat *= signs[:, None]
    return u_mat, vh_mat
