# Pytorch Principal Component Analysis (PCA)

Principal Component Anlaysis (PCA) in PyTorch. The intention is to provide a
simple and easy to use implementation of PCA in PyTorch, the most similar to
the `sklearn`'s PCA as possible (in terms of API and, of course, output).
Plus, this implementation is **fully differentiable and faster** (thanks to GPU parallelization)!

[![Release](https://img.shields.io/github/v/tag/valentingol/torch_pca?label=Pypi&logo=pypi&logoColor=yellow)](https://pypi.org/project/torch_pca/)
![PythonVersion](https://img.shields.io/badge/python-3.8%20%7E%203.11-informational)
![PytorchVersion](https://img.shields.io/badge/pytorch-1.8%20%7E%201.13%20%7C%202.0+-informational)

[![GitHub User followers](https://img.shields.io/github/followers/valentingol?label=User%20followers&style=social)](https://github.com/valentingol)
[![GitHub User's User stars](https://img.shields.io/github/stars/valentingol?label=User%20Stars&style=social)](https://github.com/valentingol)

[![Ruff_logo](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![Black_logo](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Ruff](https://github.com/valentingol/torch_pca/actions/workflows/ruff.yaml/badge.svg)](https://github.com/valentingol/Dinosor/actions/workflows/ruff.yaml)
[![Flake8](https://github.com/valentingol/torch_pca/actions/workflows/flake.yaml/badge.svg)](https://github.com/valentingol/Dinosor/actions/workflows/flake.yaml)
[![MyPy](https://github.com/valentingol/torch_pca/actions/workflows/mypy.yaml/badge.svg)](https://github.com/valentingol/Dinosor/actions/workflows/mypy.yaml)
[![PyLint](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/valentingol/8fb4f3f78584e085dd7b0cca7e046d1f/raw/torch_pca_pylint.json)](https://github.com/valentingol/torch_pca/actions/workflows/pylint.yaml)

[![Tests](https://github.com/valentingol/torch_pca/actions/workflows/tests.yaml/badge.svg)](https://github.com/valentingol/torch_pca/actions/workflows/tests.yaml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/valentingol/c5a6b5731db93da673f8e258b2669080/raw/torch_pca_tests.json)](https://github.com/valentingol/torch_pca/actions/workflows/tests.yaml)
[![Documentation Status](https://readthedocs.org/projects/torch-pca/badge/?version=latest)](https://torch-pca.readthedocs.io/en/latest/?badge=latest)

## Links

Github repository: https://github.com/valentingol/torch_pca

Pypi project: https://pypi.org/project/torch_pca/

Documentation: https://torch-pca.readthedocs.io/en/latest/

## Installation

Simply install it with pip:

```bash
pip install torch-pca
```

## How to use

Exactly like `sklearn.decomposition.PCA` but it uses PyTorch tensors as input and output!

```python
from torch_cpa import PCA

# Create like sklearn.decomposition.PCA, e.g.:
pca_model = PCA(n_components=None, svd_solver='full')

# Use like sklearn.decomposition.PCA, e.g.:
>>> new_train_data = pca_model.fit_transform(train_data)
>>> new_test_data = pca_model.transform(test_data)
>>> print(pca.explained_variance_ratio_)
[0.756, 0.142, 0.062, ...]
```

More details and features in the [API documentation](https://torch-pca.readthedocs.io/en/latest/api.html#torch_pca.pca_main.PCA).

## Gradient backward pass

Use the pytorch framework allows the automatic differentiation of the PCA!

The PCA transform method is always differentiable so it is always possible to
compute gradient like that:

```python
pca = PCA()
for ep in range(n_epochs):
    optimizer.zero_grad()
    out = neural_net(inputs)
    with torch.no_grad():
        pca.fit(out)
    out = pca.transform(out)
    loss = loss_fn(out, targets)
    loss.backward()
```

If you want to compute the gradient over the full PCA model (including the
fitted `pca.n_components`), you can do it by using the "full" SVD solver
and removing the part of the `fit` method that enforce the deterministic
output by passing `determinist=False` in `fit` or `fit_transform` method.
This part sort the components using the singular values and change their sign
accordingly so it is not differentiable by nature but may be not necessary if
you don't care about the determinism of the output:

```python
pca = PCA(svd_solver="full")
for ep in range(n_epochs):
    optimizer.zero_grad()
    out = neural_net(inputs)
    out = pca.fit_transform(out, determinist=False)
    loss = loss_fn(out, targets)
    loss.backward()
```

## Comparison of execution time with sklearn's PCA

As we can see below the PyTorch PCA is faster than sklearn's PCA, in all the
configs tested with the parameter by default (for each PCA model):

![include](docs/_static/comparison.png)

## Implemented features

- [x] `fit`, `transform`, `fit_transform` methods.
- [x] All attributes from sklean's PCA are available: `explained_variance_(ratio_)`,
      `singular_values_`, `components_`, `mean_`, `noise_variance_`, ...
- [x] Full SVD solver
- [x] SVD by covariance matrix solver
- [x] Randomized SVD solver
- [x] (absent from sklearn) Decide how to center the input data in `transform` method
  (default is like sklearn's PCA)
- [x] Find number of components with explained variance proportion
- [x] Automatically find number of components with MLE
- [x] `inverse_transform` method
- [x] Whitening option
- [x] `get_covariance` method
- [x] `get_precision` method and `score`/`score_samples` methods

## To be implemented

- [ ] Support sparse matrices with ARPACK solver

## Contributing

Feel free to contribute to this project! Just fork it and make an issue or a pull request.

See the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.
