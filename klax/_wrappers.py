"""This module implements parameter constraints based on paramax."""

from __future__ import annotations

from abc import abstractmethod
import equinox as eqx
import jax
from jaxtyping import Array, PyTree
from paramax import AbstractUnwrappable

from typing import Union, TypeVar, Generic, final, TypeAlias

# Type definition for (potentially nested) unwrappables around Jax Arrays
ParameterWrapper: TypeAlias = AbstractUnwrappable[Union[Array, "ParameterWrapper"]]

T = TypeVar("T")


class AbstractUpdatable(
    AbstractUnwrappable[Union[T, "AbstractUpdatable[T]"]], Generic[T]
):
    """An abstract class representing an updatable and unwrappable object.

    Updatables replace PyTree nodes, applying custom behavior upon updating.

    Inherrit from this class and define an :meth:`update` behavior to implement
    custom updatables.
    """

    parameter: Union[T, AbstractUpdatable[T]]

    @final
    def unwrap(self) -> T:
        """Returns the unwrapped parameter, assuming no wrapped subnodes exist."""
        return self.parameter

    @abstractmethod
    def update(self) -> T:
        """Returns the updated parameter, assuming no wrapped subnodes exist."""
        return self.parameter


# The following is inspired by paramax unwrap
def update_wrapper(tree: PyTree):
    """Map across a PyTree and update all :class:`AbstractUpdatable` nodes.

    This leaves all other nodes unchanged. If nested, the innermost
    :class:`AbstractUpdatable` nodes are updated first.

    .. note::

        The actual behavior of nested :class:`AbstractUpdatable` instances
        is complex and one should be careful when composing updatables. Even more
        care should be taken if combining updatables with more general unwrappables.
        When calling update_wrapper on a nested :class:`AbstractUpdatable`,
        the innermost parameter value is indirectly passed through
        each update function of the Updatables, starting from the innermost
        node moving outwards.

    Args:
        tree: Arbitrary pytree containing :class:`AbstractUpdatable` nodes.
    """

    def get_innermost(tree: AbstractUpdatable):
        """Retrieve the innermost parameter of a given (potentially nested)
        :class:`AbstractUpdatable` pytree"""
        p = tree.parameter
        if isinstance(p, AbstractUpdatable):
            return get_innermost(p)
        return p

    def _update_wrapper(tree, *, include_self: bool):
        def _map_fn(leaf):
            if isinstance(leaf, AbstractUpdatable):
                # Apply all subnodes
                leaf = _update_wrapper(leaf, include_self=False)

                # Assign the innermost parameter to parameter of the current leaf ...
                _leaf = eqx.tree_at(lambda t: t.parameter, leaf, get_innermost(leaf))
                # ... then call apply on the resulting AbstractUpdatable to get the updated parameter
                p = _leaf.update()
                return eqx.tree_at(get_innermost, leaf, p)
            return leaf

        def is_leaf(x):
            is_unwrappable = isinstance(x, AbstractUpdatable)
            included = include_self or x is not tree
            return is_unwrappable and included

        return jax.tree_util.tree_map(f=_map_fn, tree=tree, is_leaf=is_leaf)

    return _update_wrapper(tree, include_self=True)


class Positive(AbstractUnwrappable[Array]):
    """Wrapper that applies a non-negative constraint by passing the weight
    trough softplus.

    Args:
        parameter: The parameter that is to be made non-negative. It can either
            be a `jax.Array` or a `paramax.AbstractUnwrappable`that is wrapped
            around a `jax.Array`.
    """

    parameter: Array

    def unwrap(self) -> Array:
        return jax.nn.softplus(self.parameter)


class NonNegative(AbstractUpdatable[Array]):
    """Updatable wrapper that clamps the wrapped array elementwise onto
    the non-negative range when :func:`update_wrapper` is called on a
    pytree containing this wrapper.

    Args:
        parameter: The parameter that is to be made non-negative on update.
            It can either be a ``jax.Array`` or a ``AbstractUpdatable[Array]``.

    Example:
        .. doctest::

            >>> import klax
            >>> import jax.numpy as jnp
            >>> value = jnp.array([-1., 0., 1.])
            >>> params = ("abc", 1, klax.NonNegative(value))
            >>> params = klax.update_wrapper(params)
            >>> klax.unwrap(("abc", 1, params))
            ('abc', 1, Array([0., 0., 1.], dtype=float32))
    """

    parameter: Array

    def __init__(self, parameter: Array):
        # FIXME: This is just a quick fix of the issue that unwrap will 
        # no longer make this positive and thus freshly created FICNN will not be 
        # convex after unwraping. I suggest creating a new klax wide function:
        # klax.finalize = klax.unwrap(klax.update_wrappers).
        self.parameter = jax.nn.relu(parameter)

    def update(self) -> Array:
        return jax.nn.relu(self.parameter)
