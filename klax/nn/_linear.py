from typing import (
    Literal,
    Optional,
    Union,
)

import equinox as eqx
from jax.nn.initializers import Initializer, zeros
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .._misc import default_floating_dtype


class Linear(eqx.Module, strict=True):
    """Performs a linear transformation.

    This class is modified from eqx.nn.Linear to allow for custom initialization.
    """

    weight: Array
    bias: Optional[Array]
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
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `in_features`: The input size. The input to the layer should be a vector of
            shape `(in_features,)`
        - `out_features`: The output size. The output from the layer will be a vector
            of shape `(out_features,)`.
        - `weight_init`: The weight initializer of type `jax.nn.initializers.Initializer`.
        - `bias_init`: The bias initializer of type `jax.nn.initializers.Initializer`.
        - `use_bias`: Whether to add on a bias as well.
        - `dtype`: The dtype to use for the weight and the bias in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_features` also supports the string `"scalar"` as a special value.
        In this case the input to the layer should be of shape `()`.

        Likewise `out_features` can also be a string `"scalar"`, in which case the
        output from the layer will have shape `()`.

        Further note that, some `jax.nn.initializers.Initializer`s do not work if
        one of `in_features` or `out_features` is zero.

        Likewise, some `jax.nn.initializers.Initialzers`s do not work when `dtype` is
        `jax.numpy.complex64`.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        wkey, bkey = jrandom.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        wshape = (out_features_, in_features_)
        self.weight = weight_init(wkey, wshape, dtype)
        bshape = (out_features_,)
        self.bias = bias_init(bkey, bshape, dtype) if use_bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(in_features,)`. (Or shape
            `()` if `in_features="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        !!! info

            If you want to use higher order tensors as inputs (for example featuring "
            "batch dimensions) then use `jax.vmap`. For example, for an input `x` of "
            "shape `(batch, in_features)`, using
            ```python
            linear = equinox.nn.Linear(...)
            jax.vmap(linear)(x)
            ```
            will produce the appropriate output of shape `(batch, out_features)`.

        **Returns:**

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