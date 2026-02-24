---
title: Callbacks
---

Callbacks allow users to inject custom behavior into the
training loop *after* each parameter update. This can be
used for logging, early stopping or modifying the model in
a jax-incompatible way.

::: klax.Callback
    options:
        members:
            - on_training_start
            - on_training_step
            - on_training_end
