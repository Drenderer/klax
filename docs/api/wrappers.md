---
title: Unwrappables and Constraints
---

Klax provides a powerful framework for constraining learnable parameters. It extends [Paramax'](https://danielward27.github.io/paramax/index.html) concept of [Unwrappables][klax.Unwrappable] to non-differentiable [Constraints][klax.Constraint].

In short, a [Unwrappable][klax.Unwrappable] wraps around a model parameter (or entire subtree) and applies custom behavior upon [unwrapping][klax.unwrap]. Models are [unwrapped][klax.unwrap] inside of the [loss function][klax.Loss], ensuring that the custom unwrap-behavior influences the gradient computation, see, e.g., [NonTrainable][klax.NonTrainable]. However, non-differentiable custom behavior, such as clipping a parameter to a certain range, need extra treatment in order to avoid dead parameters (Parameters that never receive gradient-updates). Thats where [Constraints][klax.Constraint] come in. [Constraints][klax.Constraint] are [Unwrappables][klax.Unwrappable] that generally leave the parameter untouched upon [unwrapping][klax.unwrap]. However, when [applying][klax.apply] the [Constraints][klax.Constraint] the wrapped parameter is modified in-place. In the default [training loop][klax.make_step] all constraints are [applied][klax.apply] after each parameter update.

## Basic classes and functions

::: klax.Unwrappable
    options:
        members: 
            - unwrap
::: klax.unwrap
::: klax.contains_unwrappables
---
::: klax.Constraint
    options:
        members: 
            - unwrap
            - apply
::: klax.apply
::: klax.contains_constraints
---
::: klax.finalize

---

## Unwrappables

::: klax.Parameterize
    options:
        members:
            - __init__
            - __call__
::: klax.NonTrainable
    options:
        members:
            - __init__
            - __call__
::: klax.non_trainable
::: klax.Symmetric
    options:
        members:
            - __init__
            - __call__
::: klax.SkewSymmetric
    options:
        members:
            - __init__
            - __call__

---

## Constraints

::: klax.NonNegative
    options:
        members:
            - __init__
            - __call__

