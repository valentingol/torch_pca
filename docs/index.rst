Pytorch PCA
===========

Principal Component Anlaysis (PCA) in PyTorch. The intention is to
provide a simple and easy to use implementation of PCA in PyTorch, the
most similar to the ``sklearn``\ ’s PCA as possible (in terms of API
and, of course, output). Plus, this implementation is **fully differentiable and faster**
(thanks to GPU parallelization)!

|Release| |PythonVersion| |PytorchVersion|

|GitHub User followers| |GitHub User’s User stars|

|Ruff_logo| |Black_logo|

|Ruff| |Flake8| |MyPy| |PyLint|

|Tests| |Coverage| |Documentation Status|

Links
-----

Github repository: https://github.com/valentingol/torch_pca

Pypi project: https://pypi.org/project/torch_pca/

Documentation: https://torch-pca.readthedocs.io/en/latest/

.. |Release| image:: https://img.shields.io/github/v/tag/valentingol/torch_pca?label=Pypi&logo=pypi&logoColor=yellow
   :target: https://pypi.org/project/torch_pca/
.. |PythonVersion| image:: https://img.shields.io/badge/python-3.8%20%7E%203.11-informational
.. |PytorchVersion| image:: https://img.shields.io/badge/pytorch-1.8%20%7E%201.13%20%7C%202.0+-informational
.. |GitHub User followers| image:: https://img.shields.io/github/followers/valentingol?label=User%20followers&style=social
   :target: https://github.com/valentingol
.. |GitHub User’s User stars| image:: https://img.shields.io/github/stars/valentingol?label=User%20Stars&style=social
   :target: https://github.com/valentingol
.. |Ruff_logo| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json
   :target: https://github.com/charliermarsh/ruff
.. |Black_logo| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |Ruff| image:: https://github.com/valentingol/torch_pca/actions/workflows/ruff.yaml/badge.svg
   :target: https://github.com/valentingol/Dinosor/actions/workflows/ruff.yaml
.. |Flake8| image:: https://github.com/valentingol/torch_pca/actions/workflows/flake.yaml/badge.svg
   :target: https://github.com/valentingol/Dinosor/actions/workflows/flake.yaml
.. |MyPy| image:: https://github.com/valentingol/torch_pca/actions/workflows/mypy.yaml/badge.svg
   :target: https://github.com/valentingol/Dinosor/actions/workflows/mypy.yaml
.. |PyLint| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/valentingol/8fb4f3f78584e085dd7b0cca7e046d1f/raw/torch_pca_pylint.json
   :target: https://github.com/valentingol/torch_pca/actions/workflows/pylint.yaml
.. |Tests| image:: https://github.com/valentingol/torch_pca/actions/workflows/tests.yaml/badge.svg
   :target: https://github.com/valentingol/torch_pca/actions/workflows/tests.yaml
.. |Coverage| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/valentingol/c5a6b5731db93da673f8e258b2669080/raw/torch_pca_tests.json
   :target: https://github.com/valentingol/torch_pca/actions/workflows/tests.yaml
.. |Documentation Status| image:: https://readthedocs.org/projects/torch-pca/badge/?version=latest
   :target: https://torch-pca.readthedocs.io/en/latest/?badge=latest

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   howto
   grad
   api
   contributing.md
   license.md
