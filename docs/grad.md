# Gradient backward pass

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
