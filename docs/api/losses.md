---
title: Loss functions
---

Loss functions define the objective minimized during training with
[klax.fit][]. In klax, a loss receives `(model, batch, run_state)` and
returns a scalar value. Built-in options like [klax.mse][] and [klax.mae][]
cover common regression tasks, while custom losses can be created either with
the [loss][klax.loss] decorator or by implementing a custom
[Loss][klax.Loss] class.

---
## Ready to use loss functions
::: klax.mse
::: klax.mae
---
## Defining custom loss functions
To define a custom loss function use the [loss][klax.loss] decorator. Behind the scenes this converts your custom function into a [Loss][klax.Loss] object. In advanced cases, where you need finer control on how the loss or its gradient is calculated, you can also directly implement a custom [Loss][klax.Loss] object.
::: klax.loss
::: klax.Loss
    options:
        members:
            - __call__
            - value
            - value_and_grad

