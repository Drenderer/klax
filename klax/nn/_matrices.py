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

"""
Implementation on (constrained) matrix-valued functions:
A: R^n |-> R^(..., N, M)
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, he_normal, variance_scaling, zeros
from jaxtyping import Array, PRNGKeyArray

from .._misc import default_floating_dtype
from .._wrappers import (
    ContainsUnwrappables,
    SkewSymmetric,
    contains_unwrappables,
)
from . import MLP

type AtLeast2DTuple[T] = tuple[T, T, *tuple[T, ...]]


class Matrix(eqx.Module):
    """
    An unconstrained matrix-valued function based on an MLP.

    The MLP maps to a vector of elements which is transformed into a matrix.
    """

    mlp: MLP
    shape: AtLeast2DTuple[int] = eqx.field(static=True)

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        shape: int | AtLeast2DTuple[int] | None = None,
        width_sizes: Sequence[int] | None = None,
        weight_init: Initializer = he_normal(),
        bias_init: Initializer = zeros,
        activation: Callable = jax.nn.softplus,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype: Any | None = None,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            in_size: The input size. The input to the module should be a vector
                of shape `(in_size,)`
            shape: The matrix shape. The output from the module will be a
                Array with the specified `shape`. For square matrices a single
                integer N can be used as a shorthand for (N, N).
                (Defaults to `(in_size, in_size)`)
            width_sizes: The sizes of each hidden layer of the underlying MLP in a list.
                (Defaults to `[k,]`, where `k=max(8, in_size`).)
            weight_init: The weight initializer of type `jax.nn.initializers.Initializer`.
                (Defaults to `he_normal()`)
            bias_init: The bias initializer of type `jax.nn.initializers.Initializer`.
                (Defaults to `zeros`)
            activation: The activation function after each hidden layer.
                (Defaults to ReLU).
            final_activation: The activation function after the output layer.
                (Defaults to the identity.)
            use_bias: Whether to add on a bias to internal layers.
                (Defaults to `True`.)
            use_final_bias: Whether to add on a bias to the final layer.
                (Defaults to `True`.)
            dtype: The dtype to use for all the weights and biases in this MLP.
                (Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.)
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)

        Note:
            Note that `in_size` also supports the string `"scalar"` as a special
            value. In this case the input to the module should be of shape `()`.
        """
        in_size_ = 1 if in_size == "scalar" else in_size
        shape = in_size_ if shape is None else shape
        width_sizes = (
            [max(8, in_size_)] if width_sizes is None else width_sizes
        )
        shape = shape if isinstance(shape, tuple) else (shape, shape)

        out_size = int(jnp.prod(jnp.array(shape)))
        self.shape = shape
        self.mlp = MLP(
            in_size,
            out_size,
            width_sizes,
            weight_init,
            bias_init,
            activation,
            final_activation,
            use_bias,
            use_final_bias,
            dtype,
            key=key,
        )

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: The input. Should be a JAX array of shape ``(in_size,)``. (Or
                shape ``()`` if ``in_size="scalar"``.)

        Returns:
            A JAX array of shape ``shape``.
        """
        return self.mlp(x).reshape(self.shape)


class ConstantMatrix(eqx.Module):
    """A constant, unconstrained matrix.

    It is a wrapper around a constant array that implements the matrix-valued
    function interface.
    """

    array: Array
    shape: AtLeast2DTuple[int] = eqx.field(static=True)

    def __init__(
        self,
        shape: int | AtLeast2DTuple[int],
        init: Initializer = variance_scaling(
            scale=1, mode="fan_avg", distribution="normal"
        ),
        dtype: Any | None = None,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            shape: The matrix shape. The output from the module will be a
                Array with sthe specified `shape`. For square matrices a single
                integer N can be used as a shorthand for (N, N).
            init: The array initializer of type `jax.nn.initializers.Initializer`.
                (Defaults to `variance_scaling(scale=1, mode="fan_avg", distribution="normal")`.)
            dtype: The dtype to use for all the weights and biases in this MLP.
                (Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.)
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        self.shape = shape if isinstance(shape, tuple) else (shape, shape)
        self.array = init(key, self.shape, dtype)

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: Ignored; provided for compatibility with the rest of the
                Matrix-valued function API.

        Returns:
            A JAX array of shape ``shape``.
        """
        return self.array


class SkewSymmetricMatrix(eqx.Module):
    """A kkew-symmetric matrix-valued function based on an MLP.

    The MLP maps the input to a vector of elements that are transformed into a
    skew-symmetric matrix.
    """

    mlp: MLP
    shape: AtLeast2DTuple[int] = eqx.field(static=True)

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        shape: int | AtLeast2DTuple[int] | None = None,
        width_sizes: Sequence[int] | None = None,
        weight_init: Initializer = he_normal(),
        bias_init: Initializer = zeros,
        activation: Callable = jax.nn.softplus,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype: Any | None = None,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            in_size: The input size. The input to the module should be a vector
                of shape `(in_size,)`
            shape: The matrix shape. The output from the module will be a
                Array with sthe specified `shape`. For square matrices a single
                integer N can be used as a shorthand for (N, N).
                (Defaults to `(in_size, in_size)`)
            width_sizes: The sizes of each hidden layer of the underlying MLP in a list.
                (Defaults to `[k,]`, where `k=max(8, in_size`).)
            weight_init: The weight initializer of type `jax.nn.initializers.Initializer`.
                (Defaults to `he_normal()`)
            bias_init: The bias initializer of type `jax.nn.initializers.Initializer`.
                (Defaults to `zeros`)
            activation: The activation function after each hidden layer.
                (Defaults to `softplus`).
            final_activation: The activation function after the output layer.
                (Defaults to the identity.)
            use_bias: Whether to add on a bias to internal layers.
                (Defaults to `True`.)
            use_final_bias: Whether to add on a bias to the final layer.
                (Defaults to `True`.)
            dtype: The dtype to use for all the weights and biases in this MLP.
                (Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.)
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)

        Note:
            Note that `in_size` also supports the string `"scalar"` as a special
            value. In this case the input to the module should be of shape `()`.
        """
        in_size_ = 1 if in_size == "scalar" else in_size
        shape = in_size_ if shape is None else shape
        width_sizes = (
            [max(8, in_size_)] if width_sizes is None else width_sizes
        )
        shape = shape if isinstance(shape, tuple) else (shape, shape)
        if shape[-1] != shape[-2]:
            raise ValueError(
                "The last two dimensions in shape must be equal for skew-symmetric matrices."
            )

        out_size = int(jnp.prod(jnp.array(shape)))
        self.shape = shape
        self.mlp = MLP(
            in_size,
            out_size,
            width_sizes,
            weight_init,
            bias_init,
            activation,
            final_activation,
            use_bias,
            use_final_bias,
            dtype,
            key=key,
        )

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: The input. Should be a JAX array of shape ``(in_size,)``. (Or
                shape ``()`` if ``in_size="scalar"``.)

        Returns:
            A JAX array of shape ``shape``.
        """
        A = self.mlp(x).reshape(self.shape)
        return A - A.mT


class ConstantSkewSymmetricMatrix(eqx.Module):
    """A constant skew-symmetric matrix.

    It is a wrapper around a constant skew-symmetry-constraind array that
    implements the matrix-valued function interface.
    """

    array: SkewSymmetric
    shape: AtLeast2DTuple[int] = eqx.field(static=True)

    def __init__(
        self,
        shape: int | AtLeast2DTuple[int],
        init: Initializer = variance_scaling(
            scale=1, mode="fan_avg", distribution="normal"
        ),
        dtype: Any | None = None,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initializes the object

        Args:
            shape: The matrix shape. The output from the module will be a
                Array with sthe specified `shape`. For square matrices a single
                integer N can be used as a shorthand for (N, N).
            init: The array initializer of type `jax.nn.initializers.Initializer`.
                (Defaults to `variance_scaling(scale=1, mode="fan_avg", distribution="normal")`.)
            dtype: The dtype to use for all the weights and biases in this MLP.
                (Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.)
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        self.shape = shape if isinstance(shape, tuple) else (shape, shape)
        array = init(key, self.shape, dtype)
        self.array = SkewSymmetric(array)

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: Ignored; provided for compatibility with the rest of the
                Matrix-valued function API.

        Returns:
            A JAX array of shape ``shape``.
        """
        if contains_unwrappables(self):
            raise ContainsUnwrappables(
                "Model must be finalized before calling, see `klax.finalize`."
            )
        array = cast(Array, self.array)
        return array


class SPDMatrix(eqx.Module):
    """A symmetric positive definite matrix-valued function based on an MLP.

    The output vector `v` of the MLP is mapped to a matrix `B`. The module's
    output is then computed via `A=B@B*`.
    """

    mlp: MLP
    shape: AtLeast2DTuple[int] = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        shape: int | AtLeast2DTuple[int] | None = None,
        epsilon: float = 1e-6,
        width_sizes: Sequence[int] | None = None,
        weight_init: Initializer = he_normal(),
        bias_init: Initializer = zeros,
        activation: Callable = jax.nn.softplus,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype: Any | None = None,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            in_size: The input size. The input to the module should be a vector
                of shape `(in_size,)`
            shape: The matrix shape. The output from the module will be a
                Array with sthe specified `shape`. For square matrices a single
                integer N can be used as a shorthand for (N, N).
                (Defaults to `(in_size, in_size)`)
            width_sizes: The sizes of each hidden layer of the underlying MLP in a list.
                (Defaults to `[k,]`, where `k=max(8, in_size`).)
            epsilon: Small value that is added to the diagonal of the output matrix
                to ensure positive definiteness. If only positive semi-definiteness is
                required set `epsilon = 0.`
                (Defaults to `1e-6`)
            weight_init: The weight initializer of type `jax.nn.initializers.Initializer`.
                (Defaults to `he_normal()`)
            bias_init: The bias initializer of type `jax.nn.initializers.Initializer`.
                (Defaults to `zeros`)
            activation: The activation function after each hidden layer.
                (Defaults to `softplus`)
            final_activation: The activation function after the output layer.
                (Defaults to the identity.)
            use_bias: Whether to add on a bias to internal layers.
                (Defaults to `True`.)
            use_final_bias: Whether to add on a bias to the final layer.
                (Defaults to `True`.)
            dtype: The dtype to use for all the weights and biases in this MLP.
                (Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.)
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)

        Note:
            Note that `in_size` also supports the string `"scalar"` as a special
            value. In this case the input to the module should be of shape `()`.
        """
        in_size_ = 1 if in_size == "scalar" else in_size
        shape = in_size_ if shape is None else shape
        width_sizes = (
            [max(8, in_size_)] if width_sizes is None else width_sizes
        )
        shape = shape if isinstance(shape, tuple) else (shape, shape)
        if shape[-1] != shape[-2]:
            raise ValueError(
                "The last two dimensions in shape must be equal for symmetric matrices."
            )

        out_size = int(jnp.prod(jnp.array(shape)))
        self.shape = shape
        self.epsilon = epsilon
        self.mlp = MLP(
            in_size,
            out_size,
            width_sizes,
            weight_init,
            bias_init,
            activation,
            final_activation,
            use_bias,
            use_final_bias,
            dtype,
            key=key,
        )

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: The input. Should be a JAX array of shape ``(in_size,)``. (Or
                shape ``()`` if ``in_size="scalar"``.)

        Returns:
            A JAX array of shape ``shape``.
        """
        L = self.mlp(x).reshape(self.shape)
        A = L @ jnp.conjugate(L.mT)
        identity = jnp.broadcast_to(jnp.eye(self.shape[-1]), A.shape)
        return A + self.epsilon * identity


class ConstantSPDMatrix(eqx.Module):
    """A constant symmetric positive definite matrix-valued function.

    It is a wrapper around a constant symmetric postive semi-definite matrix
    with the matrix-valued function interface.
    """

    B_matrix: Array
    shape: AtLeast2DTuple[int] = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(
        self,
        shape: int | AtLeast2DTuple[int],
        epsilon: float = 1e-6,
        init: Initializer = variance_scaling(
            scale=1, mode="fan_avg", distribution="normal"
        ),
        dtype: Any | None = None,
        *,
        key: PRNGKeyArray,
    ):
        """
        Initializes the object

        Args:
            shape: The matrix shape. The output from the module will be a
                Array with sthe specified `shape`. For square matrices a single
                integer N can be used as a shorthand for (N, N).
            epsilon: Small value that is added to the diagonal of the output matrix
                to ensure positive definiteness. If only positive semi-definiteness is
                required set `epsilon = 0.`
                (Defaults to `1e-6`)
            init: The initializer of type `jax.nn.initializers.Initializer` for the
                constant matrix `B` that produces the module's output via `A = B@B*`.
                (Defaults to `variance_scaling(scale=1, mode="fan_avg", distribution="normal")`.)
            dtype: The dtype to use for all the weights and biases in this MLP.
                (Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.)
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)
        """
        shape = shape if isinstance(shape, tuple) else (shape, shape)
        if shape[-1] != shape[-2]:
            raise ValueError(
                "The last two dimensions in shape must be equal for symmetric matrices."
            )

        self.shape = shape
        self.epsilon = epsilon
        self.B_matrix = init(key, shape, dtype)

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: Ignored; provided for compatibility with the rest of the
                Matrix-valued function API.

        Returns:
            A JAX array of shape ``shape``.
        """
        A = self.B_matrix @ jnp.conjugate(self.B_matrix.mT)
        identity = jnp.broadcast_to(jnp.eye(self.shape[-1]), A.shape)
        return A + self.epsilon * identity
