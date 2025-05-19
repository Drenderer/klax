"""This module implements parameter constraints based on paramax."""

from __future__ import annotations
from abc import abstractmethod
from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
import paramax as px


# Alias to px.unwrap
unwrap = px.unwrap

class ParameterWrapper(px.AbstractUnwrappable[Array]):
    """An abstract class representing parameter wrappers.

    ``ParameterWrappers`` are a specialized version of ``paramax.AbstractUnwrappable``
    that returns an updated version of itself upon applying. ``ParameterWrappers``
    cannot be nested but are fully compatible with paramax's unwrapping functionality.
    """

    def __init__(self, parameter: Array):
        raise NotImplementedError("To be implemented by derived classes")

    @abstractmethod
    def unwrap(self) -> Array:
        pass

    @abstractmethod
    def apply(self) -> Self:
        pass


class NonNegative(ParameterWrapper):
    """Applies a non-negative constraint.

    Args:
        parameter: The ``jax.Array`` that is to be made non-negative upon unwrapping
            and applying.
    """

    parameter: Array

    def __init__(self, parameter: Array):
        # Ensure that the parameter fulfills the constraint initially
        self.parameter = self._non_neg(parameter)

    def _non_neg(self, x: Array) -> Array:
        return jnp.maximum(x, 0)

    def unwrap(self) -> Array:
        return self._non_neg(self.parameter)

    def apply(self) -> Array:
        return NonNegative(self._non_neg(self.parameter))


def apply(tree: PyTree):
    """Map across a ``PyTree`` and apply all :class:`ParameterWrapper` nodes.

    This leaves all other nodes unchanged. 
    
    Note:
        ``ParameterWrapper`` nodes cannot be nested.

    Example:
        Enforcing non-negativity.

        >>> import klax
        >>> import jax.numpy as jnp
        >>> params = klax.NonNegative(-1 * jnp.ones(3))
        >>> klax.apply(("abc", 1, params))
        ('abc', 1, Array([0., 0., 0.], dtype=float32))
    """

    def _unwrap(tree, *, include_self: bool):
        def _map_fn(leaf):
            if isinstance(leaf, ParameterWrapper):
                # Unwrap subnodes, then itself
                return _unwrap(leaf, include_self=False).apply()
            return leaf

        def is_leaf(x):
            is_unwrappable = isinstance(x, ParameterWrapper)
            included = include_self or x is not tree
            return is_unwrappable and included

        return jax.tree.map(f=_map_fn, tree=tree, is_leaf=is_leaf)

    return _unwrap(tree, include_self=True)
