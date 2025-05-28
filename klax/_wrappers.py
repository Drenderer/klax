# Copyright 2025 The Klax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file includes code from paramax (MIT License).
#
#     https://github.com/danielward27/paramax
#
# Original Copyright (c) 2022 Daniel Ward
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""``Unwrappables`` and ``Constraints`` modified and extended from paramax."""

from abc import abstractmethod
from typing import Any, Callable, Self, TypeVar

import equinox as eqx
import jax
from jax import lax
import jax.numpy as jnp
from jaxtyping import Array, PyTree


T = TypeVar("T")


# ===-----------------------------------------------------------------------===#
#  Unwrappable
# ===-----------------------------------------------------------------------===#


# This class is derived from paramax.
# Original Copyright 2022 Daniel Ward
class Unwrappable[T](eqx.Module):
    """An abstract class representing an unwrappable object.

    Unwrappables replace PyTree nodes, applying custom behavior upon unwrapping.
    This class is a renamed copy of :class:`paramax.AbstractUnwrappable`.

    Note:
        Models containing :class:`Unwrappables<Unwrappable>` need to be
        :func:`unwrapped<klax.unwrap>` or :func:`finalized<klax.finalize>`
        before they are callable.
    """

    @abstractmethod
    def unwrap(self) -> T:
        """Returns the unwrapped PyTree, assuming no wrapped subnodes exist."""
        pass


# This function is copied from paramax and has been slightly modified.
# Original Copyright 2022 Daniel Ward
def unwrap(tree: PyTree):
    """Map across a PyTree and unwrap all :class:`Unwrappables<Unwrappable>`.

    This leaves all other nodes unchanged. If nested, the innermost
    :class:`Unwrappable` is unwrapped first.

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
            if isinstance(leaf, Unwrappable):
                # Unwrap subnodes, then itself
                return _unwrap(leaf, include_self=False).unwrap()
            return leaf

        def is_leaf(x):
            is_unwrappable = isinstance(x, Unwrappable)
            included = include_self or x is not tree
            return is_unwrappable and included

        return jax.tree_util.tree_map(f=_map_fn, tree=tree, is_leaf=is_leaf)

    return _unwrap(tree, include_self=True)


# This class is derived from paramax and has been slightly modified.
# Original Copyright 2022 Daniel Ward
class Parameterize(Unwrappable[T]):
    """Unwrap an object by calling fn with args and kwargs.

    All of ``fn``, ``*args`` and ``**kwargs`` may contain trainable parameters.

    Note:
        Unwrapping typically occurs after model initialization. Therefore, if
        the :class:`Parameterize` object may be created in a vectorized context,
        we recommend ensuring that ``fn`` still unwraps correctly, e.g. by
        supporting broadcasting.

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


# This function is derived from paramax and has been substantially modified.
# Original Copyright 2022 Daniel Ward
def non_trainable(tree: PyTree):
    """Freezes parameters by wrapping inexact array or :class:`Constraint`
    leaves with :class:`NonTrainable`.

    Note:
        Regularization is likely to apply before unwrapping. To avoid
        regularization impacting non-trainable parameters, they should be
        filtered out, for example using:

        >>> eqx.partition(
        ...     ...,
        ...     is_leaf=lambda leaf: isinstance(leaf, (NonTrainable, Constraint)),
        ... )


    Wrapping the arrays in a model rather than the entire tree is often
    preferable, allowing easier access to attributes compared to wrapping the
    entire tree.

    Args:
        tree: The PyTree.
    """

    def _map_fn(leaf):
        return (
            NonTrainable(leaf)
            if eqx.is_inexact_array(leaf) or isinstance(leaf, Constraint)
            else leaf
        )

    return jax.tree.map(
        f=_map_fn,
        tree=tree,
        is_leaf=lambda x: isinstance(x, (NonTrainable, Constraint)),
    )


# This class is derived from paramax and has been lightly modified.
# Original Copyright 2022 Daniel Ward
class NonTrainable(Unwrappable[T]):
    """Applies stop gradient to all ArrayLike leaves before unwrapping.

    See also :func:`non_trainable`, which is probably a generally prefereable
    way to achieve similar behaviour, which wraps the arraylike leaves directly,
    rather than the tree. Useful to mark PyTrees (Arrays, Modules, etc.) as
    frozen/non-trainable. Note that the underlying parameters may still be
    impacted by regularization, so it is generally advised to use this as a
    suggestively named class for filtering parameters.
    """

    tree: T

    def unwrap(self) -> T:
        differentiable, static = eqx.partition(self.tree, eqx.is_array_like)
        return eqx.combine(lax.stop_gradient(differentiable), static)


class SkewSymmetric(Unwrappable[Array]):
    """Ensures skew-symmetry of a square matrix upon unwrapping.

    Warning:
        Wrapping ``SkewSymmetric`` around parameters that are
        already wrapped may lead to unexpected behavior and is
        generally discouraged.
    """

    parameter: Array

    def __init__(self, parameter: Array):
        """
        Args:
            parameter: Wrapped matrix as array of shape (..., N, N).
        """
        _array = unwrap(parameter)
        if not (_array.ndim >= 2 and _array.shape[-1] == _array.shape[-2]):
            raise ValueError(
                "Wrapped parameter must be an array of shape (..., N, N) but " \
                f"has shape {_array.shape}"
            )
        self.parameter = parameter

    def unwrap(self) -> Array:
        return self._make_skew_symmetric(self.parameter)

    @staticmethod
    def _make_skew_symmetric(x: Array) -> Array:
        return 0.5 * (x - jnp.matrix_transpose(x))


class Symmetric(Unwrappable[Array]):
    """Ensures symmetry of a square matrix upon unwrapping.

    Warning:
        Wrapping ``Symmetric`` around parameters that are
        already wrapped may lead to unexpected behavior and is
        generally discouraged.
    """

    parameter: Array

    def __init__(self, parameter: Array):
        """
        Args:
            parameter: To be wrapped matrix array of shape (..., N, N).
        """
        _array = unwrap(parameter)
        if not (_array.ndim >= 2 and _array.shape[-1] == _array.shape[-2]):
            raise ValueError(
                f"Wrapped parameter must be an array of shape (..., N, N) but has shape {_array.shape}"
            )
        self.parameter = parameter

    def unwrap(self) -> Array:
        return self._make_symmetric(self.parameter)

    @staticmethod
    def _make_symmetric(x: Array) -> Array:
        return 0.5 * (x + jnp.matrix_transpose(x))


# ===-----------------------------------------------------------------------===#
#  Constraints
# ===-----------------------------------------------------------------------===#


class Constraint(Unwrappable[Array]):
    """An abstract class derived from :class:`Unwrappable` to constrain JAX
    arrays.

    A :class:`Constraint` is an extened version of :class:`Unwrappable`, that
    marks an array in a PyTree as constrained. It implements the known
    ``unwrap`` method from :class:`Unwrappable` and adds the ``apply``
    functionality for the implementation of constraints that are
    non-differentiable or could lead to vanishing gradient during optimization. 

    We intend the following usage of the ``unwarp`` and ``apply`` methods:

    ``unwrap``: Identical functionality to an :class:`Unwrappable`.
        Use this for the implementaion of constraints that are differentiable
        and shall be applied withing the training loop. E.g., our implementation
        of :func:`fit` will unwrap the model as part of the loss function. Thus,
        the implementation of ``unwrap`` contributes to the gradients during
        training. An example would be a positivity constraint, that passes
        the array through :func:`jax.nn.softplus` upon unwrapping.

    ``apply``: New functionality added with :class:`Constraint`.
        Use this to implement non-differentiable or zero-gradient constraints
        that shall be applied `after` the parameter update and modify the
        wrapped array without unwrapping. Consequently, its suitable for the
        implementation of `non-differentiable` constraints, such as clamping
        a parameter to a range of admissible values. Apply functions should
        return a modified copy of ``Self``.
    
    Note:
        Models containing :class:`Constraints<Constraint>` need to be
        :func:`finalized<klax.finalize>` before they are callable.

    Warning:
        :class:`Constraints<Constraint>` should not be nested, as this can lead
        to unexpected behavior or errors. To combine the effects of two constriants,
        implement a custom constraint and define the combined effects via
        ``unwrap`` and ``apply``.
    """

    @abstractmethod
    def apply(self) -> Self:
        """Returns a modified copy of self. Most likely you want to use
        :func:`eqx.tree_at` for this purpose."""
        pass


def apply(tree: PyTree):
    """Map across a PyTree and apply all :class:`Constraints<Constraint>`.

    This leaves all other nodes unchanged.

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
            if isinstance(leaf, Constraint):
                # Unwrap subnodes, then itself
                return _apply(leaf, include_self=False).apply()
            return leaf

        def is_leaf(x):
            is_unwrappable = isinstance(x, Constraint)
            included = include_self or x is not tree
            return is_unwrappable and included

        return jax.tree.map(f=_map_fn, tree=tree, is_leaf=is_leaf)

    return _apply(tree, include_self=True)


class NonNegative(Constraint):
    """Applies a non-negative constraint.

    Args:
        parameter: The :class:`jax.Array` that is to be made non-negative upon
            unwrapping and applying.
    """

    parameter: Array

    @staticmethod
    def _non_neg(x: Array) -> Array:
        return jnp.maximum(x, 0)

    def unwrap(self) -> Array:
        return self.parameter

    def apply(self) -> Self:
        return eqx.tree_at(
            lambda x: x.parameter,
            self,
            replace=self._non_neg(self.parameter),
        )


# ===-----------------------------------------------------------------------===#
#  Utility functions
# ===-----------------------------------------------------------------------===#


def finalize(tree: PyTree):
    """
    Combines the functionality of :func:`apply` and :func:`unwrap`.

    Use this function to make a model containing
    :class:`Constraints<Constraint>` callable.

    Warning:
        For models/PyTrees containing :class:`Constraints<Constraint>`, only
        ``finalize`` the model after the parameter update or after
        training with :func:`fit`. This is because :func:`finalize` returns an
        unwrapped PyTree where all constraints and wrappers have been applied.
        However, this also means that the returned PyTree is no longer
        constrained.

        If you want to call a model that you want to fit afterwards, we
        recommend using a different name for the finalized model. For example::

            finalized_model = klax.finalize(model)
            y = finalzed_model(x)            # Call finalized model
            model, history = fit(model, ...) # Continue training with constrained model
    """
    return unwrap(apply(tree))


# This function contains code derived from paramax.
# Original Copyright 2022 Daniel Ward
def _tree_contains(tree: PyTree, instance_type) -> bool:
    """Check if a PyTree contains instances of ``instance_type``."""

    def _is_unwrappable(leaf):
        return isinstance(leaf, instance_type)

    leaves = jax.tree.leaves(tree, is_leaf=_is_unwrappable)
    return any(_is_unwrappable(leaf) for leaf in leaves)


# This function is derived from paramax and has been significantly modified.
# Original Copyright 2022 Daniel Ward
def contains_unwrappables(tree: PyTree) -> bool:
    """Check if a PyTree contains instances of :class:`Unwrappable`."""

    return _tree_contains(tree, Unwrappable)


def contains_constraints(tree: PyTree) -> bool:
    """Check if a PyTree contains instances of :class:`Constraint`."""

    return _tree_contains(tree, Constraint)
