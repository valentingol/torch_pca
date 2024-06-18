# How to use

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

More details and features in the [API page](https://torch-pca.readthedocs.io/en/latest/api.html#torch_pca.pca_main.PCA).
