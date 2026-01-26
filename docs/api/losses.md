---
title: Loss functions
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

