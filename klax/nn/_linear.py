from __future__ import annotations
from typing import (
    Literal,
    Optional,
    Union,
)
from collections.abc import Sequence

import equinox as eqx
from jax.nn.initializers import Initializer, zeros
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .._misc import default_floating_dtype
from ..wrappers import ParameterWrapper


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
        weight_wrap: ParameterWrapper | None = None,
        bias_wrap: ParameterWrapper | None = None,
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


class InputSplitLinear(eqx.Module, strict=True):
    """Performs a linear transformation for multiple inputs:
    `y = [W_1, W_2, ..., W_n]@[x_1, x_2, ..., x_n]^T + b`

    This layer is useful for formulating transformations with multiple
    inputs where different inputs requre different weight constraints
    or initialization for the corresponding weight matrices.
    """

    num_inputs: int
    weights: list[Array]
    bias: Optional[Array]
    in_features: tuple[Union[int, Literal["scalar"]]] = eqx.field(static=True)
    out_features: Union[int, Literal["scalar"]] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: Sequence[Union[int, Literal["scalar"]]],
        out_features: Union[int, Literal["scalar"]],
        weight_inits: Sequence[Initializer] | Initializer,
        bias_init: Initializer = zeros,
        weight_wraps: Sequence[ParameterWrapper] | ParameterWrapper | None = None,
        bias_wrap: ParameterWrapper | None = None,
        use_bias: bool = True,
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
                it is possible to apply a different inializer to each weight matrix.
                The sequence must have the same length as in_features.
            bias_init: The bias initializer of type `jax.nn.initializers.Initializer`.
            weight_wraps: An optional `klax.wrappers.ParameterWrapper` or sequence of
                `klax.wrappers.ParameterWrapper` that can be passed to enforce weight
                constraints. By specifying a sequence it is possible to apply a
                different wrapper to each weight matrix. The sequence must have the
                same length as in_features.
            bias_wrap: An optional `klax.wrappers.ParameterWrapper` that can be passed
               to enforce bias constraints.
            use_bias: Whether to add on a bias as well.
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
        num_inputs = len(in_features)
        if isinstance(weight_inits, Sequence):
            assert len(weight_inits) == num_inputs, (
                "The length of the weight_inits iterable must equal the length of in_features"
            )
        else:
            weight_inits = num_inputs * (weight_inits,)
        if isinstance(weight_wraps, Sequence):
            assert len(weight_wraps) == num_inputs, (
                "The length of the weight_wraps iterable must equal the length of in_features"
            )
        else:
            weight_wraps = num_inputs * (weight_wraps,)

        key, bkey = jrandom.split(key, 2)
        wkeys = jrandom.split(key, num_inputs)

        in_features_ = [1 if f == "scalar" else f for f in in_features]
        out_features_ = 1 if out_features == "scalar" else out_features

        wshapes = [(out_features_, i_f_) for i_f_ in in_features_]
        weights = [
            winit(wkey, wshape, dtype)
            for winit, wkey, wshape in zip(weight_inits, wkeys, wshapes)
        ]
        self.weights = [
            w if wwrap is None else wwrap(w) for w, wwrap in zip(weights, weight_wraps)
        ]

        bshape = (out_features_,)
        bias = bias_init(bkey, bshape, dtype) if use_bias else None
        self.bias = bias if bias_wrap is None else bias_wrap(bias)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.num_inputs = num_inputs

    def __call__(self, *xs: Array, key: Optional[PRNGKeyArray] = None) -> Array:
        if len(xs) != self.num_inputs:
            raise ValueError(
                f"Number of call arguments ({len(xs)}) does not match the number of inputs ({self.num_inputs})"
            )

        def mult(weight, in_feature, x):
            if in_feature == "scalar":
                if jnp.shape(x) != ():
                    raise ValueError("y must have scalar shape")
                x = jnp.broadcast_to(x, (1,))
            return weight @ x

        y = jnp.stack(
            [mult(w, f, x) for w, f, x in zip(self.weights, self.in_features, xs)],
            axis=0,
        ).sum(axis=0)
        if self.bias is not None:
            y = y + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(y) == (1,)
            y = jnp.squeeze(y)
        return y
