# Klax

A lightweight machine learning package for computational mechanics.

The package is build on top of Equinox, Optax, and Paramax. It is aimed at
providing access to implementations of specialized machine learning models, such
as custom parameter initializations, input convex neural networks (ICNNs) and
monotonic neural networks (MNNs). On top, we provide methods and classes for
customized calibration of these model.

All provided models are derived from [`equinox.Module`](https://docs.kidger.site/equinox/api/module/module/#equinox.Module) without any additional
abstractions. Hence, all models are are compatible with the JAX and Equinox
ecosystem.

All parameter constraints are derived from
[`klax.Unwrappable`][], which itself is just a [`equinox.Module`](https://docs.kidger.site/equinox/api/module/module/#equinox.Module). 
The implementation of the [`klax.Unwrappable`][] is an extension
of Paramax's [`paramax.Unwrappable`](https://danielward27.github.io/paramax/api/wrappers.html#paramax.wrappers.AbstractUnwrappable) and forms the basis for klax's non-differentiable parameter [Constraints][klax.Constraint].
