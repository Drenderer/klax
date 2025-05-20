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
    Tuple,
    Union,
)
from collections.abc import Sequence

import equinox as eqx
from jax.nn.initializers import Initializer, zeros
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .._misc import default_floating_dtype
from .._wrappers import ArrayWrapper


class Linear(eqx.Module, strict=True):
    """Performs a linear transformation.

    This class is modified from `eqx.nn.Linear` to allow for custom initialization.
    """

    weight: Array | ArrayWrapper
    bias: Optional[Array | ArrayWrapper]
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
        weight_wrap: Type[ArrayWrapper] | None = None,
        bias_wrap: Type[ArrayWrapper] | None = None,
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
            weight_warp: An optional `klax.wrappers.ArrayWrapper` that can be passed
               to enforce weight constraints.
            bias_warp: An optional `klax.wrappers.ArrayWrapper` that can be passed
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
        wshape = (in_features_, out_features_)
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
        assert not isinstance(self.weight, ArrayWrapper), (
            "Model must be unwrapped before calling."
        )
        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))
        x = jnp.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,), f"Output shape mismatch: expected (1,) for scalar output but got {jnp.shape(x)}."
            x = jnp.squeeze(x)
        return x


class InputSplitLinear(eqx.Module, strict=True):
    """Performs a linear transformation for multiple inputs `x_1, ..., x_n`:
    `y = x_1 @ W_1 + x_2 @ W_2 + ... +x_n @ W_n + b`

    This layer is useful for formulating transformations with multiple
    inputs where different inputs requre different weight constraints
    or initialization for the corresponding weight matrices.
    """

    weights: Tuple[Array | ArrayWrapper, ...]
    bias: Optional[Array | ArrayWrapper]
    in_features: Tuple[Union[int, Literal["scalar"]], ...] = eqx.field(static=True)
    out_features: Union[int, Literal["scalar"]] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    _num_inputs: int

    def __init__(
        self,
        in_features: Sequence[Union[int, Literal["scalar"]]],
        out_features: Union[int, Literal["scalar"]],
        weight_inits: Sequence[Initializer] | Initializer,
        bias_init: Initializer = zeros,
        use_bias: bool = True,
        weight_wraps: Sequence[type[ArrayWrapper] | None]
        | type[ArrayWrapper]
        | None = None,
        bias_wrap: type[ArrayWrapper] | None = None,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            in_features: The input sizes of each input. The n-th input to the
                layer should be a vectors of shape `(in_features[n],)`
            out_features: The output size. The output from the layer will be a
                vector of shape `(out_features,)`.
            weight_inits: Weight initializer or sequence of weight initializers
                of type `jax.nn.initializers.Initializer`. By specifying a sequence
                it is possible to apply a different initializer to each weight matrix.
                The sequence must have the same length as in_features.
            bias_init: The bias initializer of type `jax.nn.initializers.Initializer`.
            use_bias: Whether to add on a bias as well.
            weight_wraps: An optional `klax.wrappers.ArrayWrapper` or sequence of
                `klax.wrappers.ArrayWrapper` that can be passed to enforce weight
                constraints. By specifying a sequence it is possible to apply a
                different wrapper to each weight matrix. The sequence must have the
                same length as in_features.
            bias_wrap: An optional `klax.wrappers.ArrayWrapper` that can be passed
               to enforce bias constraints.
            dtype: The dtype to use for the weight and the bias in this layer.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)

        Note:
            Note that `in_features` also supports the
            string `"scalar"` as a special value. In this case the respective
            input to the layer should be of shape `()`.

            Likewise `out_features` can also be a string `"scalar"`, in which
            case the output from the layer will have shape `()`.

            Further note that, some `jax.nn.initializers.Initializer`s do not
            work if one of `in_features` or `out_features`
            is zero.

            Likewise, some `jax.nn.initializers.Initialzers`s do not work when
            `dtype` is `jax.numpy.complex64`.
        """
        dtype = default_floating_dtype() if dtype is None else dtype

        # Broadcast weight initializers and weight wrappers
        _num_inputs = len(in_features)
        if isinstance(weight_inits, Sequence):
            assert len(weight_inits) == _num_inputs, (
                "The length of the weight_inits is unequal to the length of in_features. "
                f"Expected length {_num_inputs} but is {len(weight_inits)}."
            )
        else:
            weight_inits = _num_inputs * (weight_inits,)

        if isinstance(weight_wraps, Sequence):
            assert len(weight_wraps) == _num_inputs, (
                "The length of the weight_wraps is unequal to the length of in_features. "
                f"Expected length {_num_inputs} but is {len(weight_wraps)}."
            )
        else:
            weight_wraps = _num_inputs * (weight_wraps,)

        key, bkey = jrandom.split(key, 2)
        wkeys = jrandom.split(key, _num_inputs)

        in_features_ = [1 if f == "scalar" else f for f in in_features]
        out_features_ = 1 if out_features == "scalar" else out_features

        wshapes = ((i_f_, out_features_) for i_f_ in in_features_)
        weights = [
            init(wkey, wshape, dtype)
            for init, wkey, wshape in zip(weight_inits, wkeys, wshapes)
        ]
        weights = [
            w if wrap is None else wrap(w) for w, wrap in zip(weights, weight_wraps)
        ]
        self.weights = tuple(weights)

        bshape = (out_features_,)
        if use_bias is None:
            self.bias = None
        else:
            bias = bias_init(bkey, bshape, dtype)
            self.bias = bias if bias_wrap is None else bias_wrap(bias)

        self.in_features = tuple(in_features)
        self.out_features = out_features
        self.use_bias = use_bias
        self._num_inputs = _num_inputs

    def __call__(self, *xs: Array, key: Optional[PRNGKeyArray] = None) -> Array:
        """
        Args:
            xs: The inputs. Should be n JAX arrays x_i of shape `(in_features[i],)`. (Or
                shape `()` if `in_features[i]="scalar"`.)
            key: Ignored; provided for compatibility with the rest of the
                Equinox API. (Keyword only argument.)

        Returns:
            A JAX array of shape `(out_features,)`. (Or shape `()` if
            `out_features="scalar"`.)
        """

        if len(xs) != self._num_inputs:
            raise ValueError(
                f"Number of call arguments ({len(xs)}) does not match the number of inputs ({self._num_inputs})"
            )

        def mult(weight, in_feature, x):
            if in_feature == "scalar":
                if jnp.shape(x) != ():
                    raise ValueError("y must have scalar shape")
                x = jnp.broadcast_to(x, (1,))
            return jnp.matmul(x, weight)

        y = jnp.stack(
            [mult(w, f, x) for w, f, x in zip(self.weights, self.in_features, xs)],
            axis=0,
        ).sum(axis=0)
        if self.bias is not None:
            y = y + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(y) == (1,), f"Output shape mismatch: expected (1,) for scalar output but got {jnp.shape(y)}."
            y = jnp.squeeze(y)
        return y
