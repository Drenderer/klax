# Klax

A lightweight machine learning package for computational mechanics built on JAX.

---

!!! warning

    Klax is still in early development and will likely see significant API changes in the near future. Likewise, the documentation is still under heavy development.

## Overview

Klax provides specialized machine learning architectures, constraints, and training utilities for mechanical engineering and physics applications. Built on top of [JAX](https://docs.jax.dev/en/latest/), [Equinox](https://docs.kidger.site/equinox/), and [Optax](https://optax.readthedocs.io/en/latest/), it offers:

- **Neural Networks**: Implementations of Input Convex Neural Networks (ICNNs), matrix-valued neural networks, MLPs with custom initialization, and more.
- **JAX Compatibility**: Seamless integration with JAX's automatic differentiation and acceleration
- **Parameter Constraints**: Differentiable and non-differentiable parameter constraints through [`klax.Unwrappable`][] and [`klax.Constraint`][]
- **Customizable Training**: Methods and APIs for customized calibrations on arbitrary PyTree data structures through [`klax.fit`][], [`klax.Loss`][], and [`klax.Callback`][]

Klax is designed to be minimally intrusive - all models inherit directly from [`equinox.Module`](https://docs.kidger.site/equinox/api/module/module/#equinox.Module) without additional abstraction layers. This ensures full compatibility with the JAX/Equinox ecosystem while adding mechanical engineering-specific functionality.

The constraint system is derived from Paramax's [`paramax.AbstractUnwrappable`](https://danielward27.github.io/paramax/api/wrappers.html#paramax.wrappers.AbstractUnwrappable), extending it to support non-differentiable parameter constraints such as ReLU-based non-negativity constraints.

The provided calibration utilities ([`klax.fit`][], [`klax.Loss`][], [`klax.Callback`][]) are designed to operate on arbitrarily shaped PyTrees of data, fully utilizing the flexibility of JAX and Equinox. While they cover most common machine learning use cases, as well as our specialized requirements, they remain entirely optional. The core building blocks of Klax work seamlessly in custom training loops.

## Installation

Klax can be installed via pip using

```bash
pip install klax
```

If you want to add the latest release of klax to your Python [uv](https://docs.astral.sh/uv/) project run

```bash
uv add klax
```

**or** get the most recent changes from the main branch via

```bash
uv add "klax @ git+https://github.com/Drenderer/klax.git@main"
```

## Getting Started

As the contents of the documentation are still rudimentary, we recommend checking out [Equinox](https://docs.kidger.site/equinox/) and [Paramax](https://danielward27.github.io/paramax/#) and taking a look at our examples [Examples](./examples/isotropic_hyperelasticity.ipynb).

## Citation

## Acknowledgement

Klax is built on top of several powerful frameworks:

[JAX](https://docs.jax.dev/en/latest/) - For automatic differentiation and acceleration </br>
[Equinox](https://docs.kidger.site/equinox/) - For neural network primitives </br>
[Optax](https://optax.readthedocs.io/en/latest/) - For optimization utilities </br>
[Paramax](https://https://danielward27.github.io/paramax/#) - For constraints (We decided to embed Paramax directly into Klax due to the need for non-differentiable constraints).
