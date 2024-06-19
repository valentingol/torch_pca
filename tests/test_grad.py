"""Test gradient backward through PCA."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
import torch
from torch import nn

from torch_pca import PCA


def test_grad_transform() -> None:
    """Test backward in transform."""
    X = torch.randn(100, 32)
    y = X[:, :10].sum(dim=1) / 10.0
    model1 = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 10))
    model2 = nn.Sequential(nn.Linear(5, 1))
    pca = PCA(n_components=5, svd_solver="full")
    optimizer = torch.optim.Adam(
        list(model1.parameters()) + list(model2.parameters()), lr=0.01
    )
    criterion = nn.MSELoss()
    for _ in range(10):
        optimizer.zero_grad()
        out = model1(X)
        with torch.no_grad():
            pca.fit(out, determinist=True)
        out_pca = pca.transform(out)
        y_pred = model2(out_pca)
        loss = criterion(y_pred, y.view(-1, 1))
        loss.backward()
        optimizer.step()


def test_grad_fit_transform() -> None:
    """Test backward in fit_transform."""
    X = torch.randn(100, 32)
    y = X[:, :10].sum(dim=1) / 10.0
    model1 = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 10))
    model2 = nn.Sequential(nn.Linear(5, 1))
    pca = PCA(n_components=5, svd_solver="full")
    optimizer = torch.optim.Adam(
        list(model1.parameters()) + list(model2.parameters()), lr=0.01
    )
    criterion = nn.MSELoss()
    for _ in range(10):
        optimizer.zero_grad()
        out = model1(X)
        out_pca = pca.fit_transform(out, determinist=False)
        y_pred = model2(out_pca)
        loss = criterion(y_pred, y.view(-1, 1))
        loss.backward()
        optimizer.step()
