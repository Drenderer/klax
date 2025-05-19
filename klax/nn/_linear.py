# This file includes code from Equinox
#
#     https://github.com/patrick-kidger/equinox
#
# licensed under Apache 2.0. Changes were made to class `Linear`.
#
# Modifications copyright 2025 The Klax Authors.
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

from __future__ import annotations
from typing import (
    Literal,
    Optional,
    Type,
    Union,
)

import equinox as eqx
from jax.nn.initializers import Initializer, zeros
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .._misc import default_floating_dtype
from .._wrappers import ParameterWrapper


class Linear(eqx.Module, strict=True):
    """Performs a linear transformation.

    This class is modified from eqx.nn.Linear to allow for custom initialization.
    """

    weight: Array | ParameterWrapper
    bias: Optional[Array | ParameterWrapper]
    in_features: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_features: Union[int, Literal["scalar"]] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        weight_init: Initializer,
        bias_init: Initializer = zeros,
        use_bias: bool = True,
        weight_wrap: Type[ParameterWrapper] | None = None,
        bias_wrap: Type[ParameterWrapper] | None = None,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            in_features: The input size. The input to the layer should be a vector of
               shape `(in_features,)`
            out_features: The output size. The output from the layer will be a vector
               of shape `(out_features,)`.
            weight_init: The weight initializer of type `jax.nn.initializers.Initializer`.
            bias_init: The bias initializer of type `jax.nn.initializers.Initializer`.
            use_bias: Whether to add on a bias as well.
            weight_warp: An optional `klax.wrappers.ParameterWrapper` that can be passed
               to enforce weight constraints.
            bias_warp: An optional `klax.wrappers.ParameterWrapper` that can be passed
               to enforce bias constraints.
            dtype: The dtype to use for the weight and the bias in this layer.
               Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
               on whether JAX is in 64-bit mode.
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
               initialisation. (Keyword only argument.)

        Note:
            Note that `in_features` also supports the string `"scalar"` as a
            special value. In this case the input to the layer should be of
            shape `()`.

            Likewise `out_features` can also be a string `"scalar"`, in which
            case the output from the layer will have shape `()`.

            Further note that, some `jax.nn.initializers.Initializer`s do not
            work if one of `in_features` or `out_features` is zero.

            Likewise, some `jax.nn.initializers.Initialzers`s do not work when
            `dtype` is `jax.numpy.complex64`.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        wkey, bkey = jrandom.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        wshape = (out_features_, in_features_)
        weight = weight_init(wkey, wshape, dtype)
        self.weight = weight if weight_wrap is None else weight_wrap(weight)
        bshape = (out_features_,)
        if use_bias is None:
            self.bias = None
        else:
            bias = bias_init(bkey, bshape, dtype)
            self.bias = bias if bias_wrap is None else bias_wrap(bias)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """
        Args:
            x: The input. Should be a JAX array of shape `(in_features,)`. (Or
                shape `()` if `in_features="scalar"`.)
            key: Ignored; provided for compatibility with the rest of the
                Equinox API. (Keyword only argument.)

        Note:
            If you want to use higher order tensors as inputs (for example featuring "
            "batch dimensions) then use `jax.vmap`. For example, for an input `x` of "
            "shape `(batch, in_features)`, using

            >>> import jax
            >>> from jax.nn.initializers import he_normal
            >>> import jax.random as jrandom
            >>> import klax
            >>>
            >>> key = jrandom.PRNGKey(0)
            >>> keys = jrandom.split(key)
            >>> x = jrandom.uniform(keys[0], (10,))
            >>> linear = klax.nn.Linear("scalar", "scalar", he_normal(), key=keys[1])
            >>> jax.vmap(linear)(x).shape
            (10,)

            will produce the appropriate output of shape `(batch, out_features)`.

        Returns:
            A JAX array of shape `(out_features,)`. (Or shape `()` if
            `out_features="scalar"`.)
        """

        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))
        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return x


class FullyLinear(eqx.Module, strict=True):
    """Performs a linear transformation for two inputs.

    This layer is useful for formulating, e.g., fully connected multi layer
    perceptrons (MLP), where the MLP input is passed as an additional input
    to each hidden layer, alongside the output of the previous layer.
    """

    weight_y: Array
    weight_z: Array
    bias: Optional[Array]
    in_features_y: Union[int, Literal["scalar"]] = eqx.field(static=True)
    in_features_z: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_features: Union[int, Literal["scalar"]] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features_y: Union[int, Literal["scalar"]],
        in_features_z: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        weight_init: Initializer,
        bias_init: Initializer = zeros,
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            in_features_y: The input size of the first input. The input to the
                layer should be a vector of shape `(in_features_y,)`
            in_features_z: The input size of the second input. The input to the
                layer should be a vector of shape `(in_features_z,)`
            out_features: The output size. The output from the layer will be a
                vector of shape `(out_features,)`.
            weight_init: The weight initializer of type `jax.nn.initializers.Initializer`.
            bias_init: The bias initializer of type `jax.nn.initializers.Initializer`.
            use_bias: Whether to add on a bias as well.
            dtype: The dtype to use for the weight and the bias in this layer.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)

        Note:
            Note that `in_features_y` and `in_features_z` also supports the
            string `"scalar"` as a special value. In this case the respective
            input to the layer should be of shape `()`.

            Likewise `out_features` can also be a string `"scalar"`, in which
            case the output from the layer will have shape `()`.

            Further note that, some `jax.nn.initializers.Initializer`s do not
            work if one of `in_features_y`, `in_features_z`, or `out_features`
            is zero.

            Likewise, some `jax.nn.initializers.Initialzers`s do not work when
            `dtype` is `jax.numpy.complex64`.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        wykey, wzkey, bkey = jrandom.split(key, 3)
        in_features_y_ = 1 if in_features_y == "scalar" else in_features_y
        in_features_z_ = 1 if in_features_z == "scalar" else in_features_z
        out_features_ = 1 if out_features == "scalar" else out_features
        wshape_y = (out_features_, in_features_y_)
        self.weight_y = weight_init(wykey, wshape_y, dtype)
        wshape_z = (out_features_, in_features_z_)
        self.weight_z = weight_init(wzkey, wshape_z, dtype)
        bshape = (out_features_,)
        self.bias = bias_init(bkey, bshape, dtype) if use_bias else None

        self.in_features_y = in_features_y
        self.in_features_z = in_features_z
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(
        self, y: Array, z: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        """
        Args:
            y: The first input. Should be a JAX array of shape `(in_features_y,)`.
                (Or shape `()` if `in_features="scalar"`.)
            z: The first input. Should be a JAX array of shape `(in_features_z,)`.
                (Or shape `()` if `in_features="scalar"`.)
            key: Ignored; provided for compatibility with the rest of the Equinox API.
                (Keyword only argument.)

        Note:
            If you want to use higher order tensors as inputs (for example featuring
            batch dimensions) then use `jax.vmap`. For example, for inputs `y` and
            `z` of shape `(batch, in_features_y)` and `(batch, in_features_z)`,
            respectively, using

            >>> import jax
            >>> from jax.nn.initializers import he_normal
            >>> import jax.random as jrandom
            >>> import klax
            >>>
            >>> key = jrandom.PRNGKey(0)
            >>> keys = jrandom.split(key, 3)
            >>> y = jrandom.uniform(keys[0], (10,))
            >>> z = jrandom.uniform(keys[1], (10,))
            >>> fully_linear = klax.nn.FullyLinear(
            ...     "scalar", "scalar", "scalar", he_normal(), key=keys[2]
            ... )
            >>> jax.vmap(fully_linear, (0, 0))(y, z).shape
            (10,)

            will produce the appropriate output of shape `(batch, out_features)`.

        Returns:
            A JAX array of shape `(out_features,)` (or shape `()` if
            `out_features="scalar").
        """

        if self.in_features_y == "scalar":
            if jnp.shape(y) != ():
                raise ValueError("y must have scalar shape")
            y = jnp.broadcast_to(y, (1,))
        if self.in_features_z == "scalar":
            if jnp.shape(z) != ():
                raise ValueError("z must have scalar shape")
            z = jnp.broadcast_to(z, (1,))
        x = self.weight_y @ y + self.weight_z @ z
        if self.bias is not None:
            x = x + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return x
