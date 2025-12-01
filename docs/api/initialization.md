---
title: Parameter initialization
---

Specialized parameter initializers, extending `jax.nn.initializers`.

 For some initialization schemes, the bias depends on the number of input features, which cannot be determined from the shape of the bias array (see, e.g, [Hoedt normal initializer](https://arxiv.org/abs/2312.12474)). To handle such cases klax provides a custom [`klax.Initializer`]() protocol that generalizes upon the JAX API, while ensuring compatibility with all [`jax.nn.initializers`](https://docs.jax.dev/en/latest/jax.nn.initializers.html).


::: klax.Initializer
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
