---
title: Loss functions
---

Loss functions define the objective minimized during training with
[klax.fit][]. In klax, a loss receives `(model, batch, run_state)` and
returns a scalar value and a dictionary mapping strings (typically
metric names) to metric values. 
Built-in options like `klax.mse` and `klax.mae` cover common regression tasks,
while custom losses can be created either with
the [loss][klax.loss] decorator or by implementing a custom
[Loss][klax.Loss] class.

---
## Ready to use loss functions

### `klax.mse`

Mean squared error between model predictions and target values. The model is
evaluated independently for each input in the batch, and the squared errors are
averaged over all batch and output elements.

```python
klax.mse(model, batch, run_state) -> (value, metrics)
```

`batch` is a pair `(x, y)`, where `x` contains the inputs and `y` contains the
corresponding targets. The leading batch dimensions must match, and the
prediction and target shapes should match so each prediction is compared with
its intended target. `run_state` is accepted to follow the loss interface but
is not used by this loss.

For `y_pred = jax.vmap(model)(x)`, the returned loss is
$\operatorname{mean}((y_{pred} - y)^2)$. The second returned value is a metrics
dictionary with the entry `"mse"` containing the same scalar as `value`.

### `klax.mae`

Mean absolute error between model predictions and target values. The model is
evaluated independently for each input in the batch, and the absolute errors
are averaged over all batch and output elements.

```python
klax.mae(model, batch, run_state) -> (value, metrics)
```

`batch` is a pair `(x, y)`, where `x` contains the inputs and `y` contains the
corresponding targets. The leading batch dimensions must match, and the
prediction and target shapes should match so each prediction is compared with
its intended target. `run_state` is accepted to follow the loss interface but
is not used by this loss.

For `y_pred = jax.vmap(model)(x)`, the returned loss is
$\operatorname{mean}(|y_{pred} - y|)$. The second returned value is a metrics
dictionary with the entry `"mae"` containing the same scalar as `value`.

---
## Defining custom loss functions
To define a custom loss function use the [`loss`][klax.loss] decorator. 
Behind the scenes this converts your custom function into a [`Loss`][klax.Loss] 
object. In advanced cases, where you need finer control on how the loss or its 
gradient is calculated, you can also directly implement a custom 
[Loss][klax.Loss] object.

::: klax.loss
::: klax.Loss
    options:
        members:
            - __call__
            - value
            - value_and_grad

