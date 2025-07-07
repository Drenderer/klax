<div align="center">
<img src="docs/assets/logo.png" alt="logo" width="200">
</div>

A lightweight machine learning package for computational mechanics built on JAX.

---

Check out the [Documentation](https://drenderer.github.io/klax/) for examples and reference material.


## What is Klax?

Klax provides specialized machine learning architectures, constraints, and training utilities for mechanics and physics applications. Built on top of [JAX](https://docs.jax.dev/en/latest/), [Equinox](https://docs.kidger.site/equinox/), and [Optax](https://optax.readthedocs.io/en/latest/), it offers:

- **Special Neural Networks**: Implementations of, e.g.,  Input Convex Neural Networks (ICNNs), matrix-valued neural networks, MLPs with custom initialization, and more.
- **JAX Compatibility**: Seamless integration with JAX's automatic differentiation and acceleration.
- **Parameter Constraints**: Differentiable and non-differentiable parameter constraints through [`klax.Unwrappable`](https://drenderer.github.io/klax/api/wrappers/#klax.Unwrappable) and [`klax.Constraint`](https://drenderer.github.io/klax/api/wrappers/#klax.Constraint)
- **Customizable Training**: Methods and APIs for customized calibrations on arbitrary PyTree data structures through [`klax.fit`](https://drenderer.github.io/klax/api/training/#klax.fit), [`klax.Loss`](https://drenderer.github.io/klax/api/losses/#klax.Loss), and [`klax.Callback`](https://drenderer.github.io/klax/api/callbacks/#klax.Callback).

Klax is designed to be minimally intrusive - all models inherit directly from [`equinox.Module`](https://docs.kidger.site/equinox/api/module/module/#equinox.Module) without additional abstraction layers. This ensures full compatibility with the JAX/Equinox ecosystem.

The constraint system is derived from Paramax's [`paramax.AbstractUnwrappable`](https://danielward27.github.io/paramax/api/wrappers.html#paramax.wrappers.AbstractUnwrappable), extending it to support non-differentiable/zero-gradient parameter constraints such as ReLU-based non-negativity constraints.

The provided calibration utilities ([`klax.fit`](https://drenderer.github.io/klax/api/training/#klax.fit), [`klax.Loss`](https://drenderer.github.io/klax/api/losses/#klax.Loss), [`klax.Callback`](https://drenderer.github.io/klax/api/callbacks/#klax.Callback)) are designed to operate on arbitrarily shaped PyTrees of data, fully utilizing the flexibility of JAX and Equinox. While they cover most common machine learning use cases, as well as our specialized requirements, they remain entirely optional. The core building blocks of Klax work seamlessly in custom training loops.

Currently Klax's training utilities are built around Optax, but different optimization libraries could be supported in the future if desired.

If you like using Klax, feel free to leave a GitHub star, and if there is a machine learning architecture that you think should be included in Klax, please consider making a PR.


## Installation

Klax requires python 3.12+.

```bash
pip install klax
```

**or** get the most recent changes from the main branch via

```bash
pip install "klax @ git+https://github.com/Drenderer/klax.git@main"
```
