"""Unwrappables and array and wrappers derived from paramax."""

from abc import abstractmethod
from typing import Any, Callable, Self, TypeVar

import equinox as eqx
import jax
from jax import lax
import jax.numpy as jnp
from jaxtyping import Array, PyTree


T = TypeVar("T")


# ===-----------------------------------------------------------------------===#
#  AbstractUnwrappable
# ===-----------------------------------------------------------------------===#


class AbstractUnwrappable[T](eqx.Module):
    """An abstract class representing an unwrappable object.

    Unwrappables replace PyTree nodes, applying custom behavior upon unwrapping.
    """

    @abstractmethod
    def unwrap(self) -> T:
        """Returns the unwrapped pytree, assuming no wrapped subnodes exist."""
        pass


def unwrap(tree: PyTree):
    """Map across a PyTree and unwrap all :class:`AbstractUnwrappable` nodes.

    This leaves all other nodes unchanged. If nested, the innermost
    ``AbstractUnwrappable`` nodes are unwrapped first.

    Example:
        Enforcing positivity.

        >>> import klax
        >>> import jax.numpy as jnp
        >>> params = klax.Parameterize(jnp.exp, jnp.zeros(3))
        >>> klax.unwrap(("abc", 1, params))
        ('abc', 1, Array([1., 1., 1.], dtype=float32))
    """

    def _unwrap(tree, *, include_self: bool):
        def _map_fn(leaf):
            if isinstance(leaf, AbstractUnwrappable):
                # Unwrap subnodes, then itself
                return _unwrap(leaf, include_self=False).unwrap()
            return leaf

        def is_leaf(x):
            is_unwrappable = isinstance(x, AbstractUnwrappable)
            included = include_self or x is not tree
            return is_unwrappable and included

        return jax.tree_util.tree_map(f=_map_fn, tree=tree, is_leaf=is_leaf)

    return _unwrap(tree, include_self=True)


class Parameterize(AbstractUnwrappable[T]):
    """Unwrap an object by calling fn with args and kwargs.

    All of fn, args and kwargs may contain trainable parameters.

    Note:

        Unwrapping typically occurs after model initialization. Therefore, if the
        ``Parameterize`` object may be created in a vectorized context, we recommend
        ensuring that ``fn`` still unwraps correctly, e.g. by supporting broadcasting.

    Example:

        >>> from klax import Parameterize, unwrap
        >>> import jax.numpy as jnp
        >>> positive = Parameterize(jnp.exp, jnp.zeros(3))
        >>> unwrap(positive)  # Aplies exp on unwrapping
        Array([1., 1., 1.], dtype=float32)

    Args:
        fn: Callable to call with args, and kwargs.
        *args: Positional arguments to pass to fn.
        **kwargs: Keyword arguments to pass to fn.
    """

    fn: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, fn: Callable, *args, **kwargs):
        self.fn = fn
        self.args = tuple(args)
        self.kwargs = kwargs

    def unwrap(self) -> T:
        return self.fn(*self.args, **self.kwargs)


def non_trainable(tree: PyTree):
    """Freezes parameters by wrapping inexact array or ``ArrayWrapper`` leaves with
    :class:`NonTrainable`.

    Note:

        Regularization is likely to apply before unwrapping. To avoid regularization
        impacting non-trainable parameters, they should be filtered out,
        for example using:

        >>> eqx.partition(
        ...     ...,
        ...     is_leaf=lambda leaf: isinstance(leaf, wrappers.NonTrainable),
        ... )


    Wrapping the arrays in a model rather than the entire tree is often preferable,
    allowing easier access to attributes compared to wrapping the entire tree.

    Args:
        tree: The pytree.
    """

    def _map_fn(leaf):
        return (
            NonTrainable(leaf)
            if eqx.is_inexact_array(leaf) or isinstance(leaf, ArrayWrapper)
            else leaf
        )

    return jax.tree.map(
        f=_map_fn,
        tree=tree,
        is_leaf=lambda x: isinstance(x, NonTrainable),
    )


class NonTrainable(AbstractUnwrappable[T]):
    """Applies stop gradient to all arraylike leaves before unwrapping.

    See also :func:`non_trainable`, which is probably a generally prefereable way to
    achieve similar behaviour, which wraps the arraylike leaves directly, rather than
    the tree. Useful to mark pytrees (arrays, submodules, etc) as frozen/non-trainable.
    Note that the underlying parameters may still be impacted by regularization,
    so it is generally advised to use this as a suggestively named class
    for filtering parameters.
    """

    tree: T

    def unwrap(self) -> T:
        differentiable, static = eqx.partition(self.tree, eqx.is_array_like)
        return eqx.combine(lax.stop_gradient(differentiable), static)


# ===-----------------------------------------------------------------------===#
#  ArrayWrapper
# ===-----------------------------------------------------------------------===#


class ArrayWrapper(AbstractUnwrappable[Array]):
    """An abstract class representing array wrappers.

    ``ArrayWrapper`` is a specialized version of ``paramax.AbstractUnwrappable``
    that returns an updated version of itself upon applying. ``ArrayWrapper``
    cannot be nested but is fully compatible with paramax's unwrapping functionality.
    """

    @abstractmethod
    def apply(self) -> Self:
        pass


def apply(tree: PyTree):
    """Map across a ``PyTree`` and apply all :class:`ArrayWrapper` nodes.

    This leaves all other nodes unchanged.

    Note:
        ``ArrayWrapper`` nodes cannot be nested.

    Example:
        Enforcing non-negativity.

        >>> import klax
        >>> import jax.numpy as jnp
        >>> params = klax.NonNegative(-1 * jnp.ones(3))
        >>> klax.apply(("abc", 1, params))
        ('abc', 1, Array([0., 0., 0.], dtype=float32))
    """

    def _apply(tree, *, include_self: bool):
        def _map_fn(leaf):
            if isinstance(leaf, ArrayWrapper):
                # Unwrap subnodes, then itself
                return _apply(leaf, include_self=False).apply()
            return leaf

        def is_leaf(x):
            is_unwrappable = isinstance(x, ArrayWrapper)
            included = include_self or x is not tree
            return is_unwrappable and included

        return jax.tree.map(f=_map_fn, tree=tree, is_leaf=is_leaf)

    return _apply(tree, include_self=True)


class NonNegative(ArrayWrapper):
    """Applies a non-negative constraint.

    Args:
        parameter: The ``jax.Array`` that is to be made non-negative upon unwrapping
            and applying.
    """

    parameter: Array

    def _non_neg(self, x: Array) -> Array:
        return jnp.maximum(x, 0)

    def unwrap(self) -> Array:
        return self._non_neg(self.parameter)

    def apply(self) -> Self:
        return eqx.tree_at(
            lambda x: x.parameter,
            self,
            replace_fn=lambda x: self._non_neg(x),
        )


# ===-----------------------------------------------------------------------===#
#  Utility functions
# ===-----------------------------------------------------------------------===#


def contains_unwrappables(pytree):
    """Check if a ``PyTree`` contains unwrappables."""

    def _is_unwrappable(leaf):
        return isinstance(leaf, AbstractUnwrappable)

    leaves = jax.tree.leaves(pytree, is_leaf=_is_unwrappable)
    return any(_is_unwrappable(leaf) for leaf in leaves)
