"""This module implements parameter constraints based on paramax."""

from __future__ import annotations

import equinox as eqx
import jax
from jaxtyping import Array, PyTree
from paramax import AbstractUnwrappable

from typing import Union


class ParameterWrapper(AbstractUnwrappable[Union[Array, "ParameterWrapper"]]):
    """Base class for (potentially nested) specialized wrappers around parameters.
    Each ``ParameterWrapper`` has a ``parameter`` attribute and
    two special methods that determine the wrappers behavior if
    :func:`klax.unwrap` or :func:`klax.apply`
    is called on a pytree that contains the ``ParameterWrapper``.
    Consider a pytree ``T`` that contains a ``ParameterWrapper`` ``P``:
    - ``unwrap``: The return value of ``P.unwrap`` replaces the
    ``P`` subtree when :func:`klax.unwrap` is called on ``T``.
    - ``apply``: The return value of ``P.apply`` replaces
    ``P.parameter`` when :func:`klax.apply` is called on ``T``.
    """

    parameter: Array | ParameterWrapper

    def unwrap(self) -> Array:
        """Returns the unwrapped pytree, assuming no wrapped subnodes exist."""
        return self.parameter

    def apply(self) -> Array | ParameterWrapper:
        """Returns the applied pytree, assuming no wrapped subnodes exist."""
        return self.parameter

    def _apply(self) -> ParameterWrapper:
        return eqx.tree_at(lambda tree: tree.parameter, self, self.apply())


# The following is inspired by paramax unwrap
def apply(tree: PyTree):
    # FIXME: This code currently cannot handle nested wrappers. That would require chaining all applies on the innermost parameter.
    """Map across a PyTree and apply all :class:`ParameterWrapper` nodes.

    This leaves all other nodes unchanged. If nested, the innermost
    ``ParameterWrapper`` nodes are applied first.
    """

    def _apply(tree, *, include_self: bool):
        def _map_fn(leaf):
            if isinstance(leaf, ParameterWrapper):
                # Unwrap subnodes, then itself
                return _apply(leaf, include_self=False)._apply()
            return leaf

        def is_leaf(x):
            is_unwrappable = isinstance(x, ParameterWrapper)
            included = include_self or x is not tree
            return is_unwrappable and included

        return jax.tree_util.tree_map(f=_map_fn, tree=tree, is_leaf=is_leaf)

    return _apply(tree, include_self=True)


class Positive(ParameterWrapper):
    """Applies a non-negative constraint by passing the weight
    trough softplus.

    Args:
        parameter: The parameter that is to be made non-negative. It can either
            be a `jax.Array` or a `paramax.AbstractUnwrappable`that is wrapped
            around a `jax.Array`.
    """

    parameter: Array

    def unwrap(self) -> Array:
        return jax.nn.softplus(self.parameter)


class NonNegative(ParameterWrapper):
    """Applies a non-negative constraint by passing the weight
    trough softplus.

    Args:
        parameter: The parameter that is to be made non-negative. It can either
            be a `jax.Array` or a `paramax.AbstractUnwrappable`that is wrapped
            around a `jax.Array`.
    """

    parameter: Array

    def apply(self) -> Array:
        return jax.nn.relu(self.parameter)
