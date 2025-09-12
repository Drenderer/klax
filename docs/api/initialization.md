---
title: Parameter initialization
---

 For some initialization schemes, the bias depends on the number of input features, which cannot be determined from the shape of the bias array. To handle such cases Klax provides a initialization framework that slightly generalizes upon JAX, while still ensuring compatibility of all klax models with the [`jax.nn.initializers`](https://docs.jax.dev/en/latest/jax.nn.initializers.html).


::: klax.KlaxInitializer
    options:
        members:
            - __call__

---

::: klax.hoedt_normal
    options:
        members:
            - __call__

---

::: klax.hoedt_bias
    options:
        members:
            - __call__