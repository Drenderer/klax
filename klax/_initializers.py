import inspect
import typing
from collections.abc import Sequence
from typing import Any, Protocol, cast

from jax import numpy as jnp
from jax import random as jr
from jax.nn.initializers import Initializer as JaxInitializer
from jaxtyping import Array, PRNGKeyArray

# Types from JAX
DTypeLikeInexact = Any
Shape = Sequence[int | Any]


@typing.runtime_checkable
class KlaxInitializer(Protocol):
    """Protocol for initializers, generalizing `jax.nn.initializers`.

    Some advanced initialization schemes initialize the bias
    depending on the number of input features (`fan_in`). However,
    from the bias shape alone `fan_in` cannot be computed. This
    Protocol specifies a initializer that is supplied with
    `fan_in` explicitly, enabeling advanced bias initialization.
    """

    def __call__(
        self,
        key: PRNGKeyArray,
        shape: Shape,
        fan_in: int,
        dtype: DTypeLikeInexact = jnp.float_,
    ) -> Array:
        raise NotImplementedError


type Initializer = JaxInitializer | KlaxInitializer


def canonicalize_initializer(init: Initializer) -> KlaxInitializer:
    """Convert `Initializer` to `KlaxInitilizer`.

    Args:
        init: Initializer (`JaxInitializer` or `KlaxInitializer`)

    Raises:
        TypeError: If the initializer call signature cannot be inspected.

    Returns:
        KlaxInitilizer

    """
    try:
        sig = inspect.signature(init)
    except ValueError:
        raise TypeError(
            "Cannot inspect this initializer; must be a Python callable with inspectable signature"
        )

    if "fan_in" in sig.parameters:
        return cast(KlaxInitializer, init)
    else:
        return lambda key, shape, fan_in, dtype=jnp.float_: init(
            key, shape, dtype
        )


def hoedt_normal(
    in_axis: int = -2,
    dtype: DTypeLikeInexact = jnp.float_,
) -> Initializer:
    def init(
        key: PRNGKeyArray, shape: Shape, dtype: DTypeLikeInexact = dtype
    ) -> Array:
        fan_in = shape[in_axis]
        mean_square = 3.440115731272907 / (
            fan_in * (1.3450928843923602 + fan_in)
        )
        variance = 1 / fan_in
        _temp_1 = jnp.log(mean_square)
        _temp_2 = jnp.log(variance + mean_square)
        mean_tilde = _temp_1 - 0.5 * _temp_2
        variance_tilde = _temp_2 - _temp_1
        weight_tilde = mean_tilde + jr.normal(key, shape, dtype) * jnp.sqrt(
            variance_tilde
        )
        return jnp.exp(weight_tilde)

    return init


def hoedt_bias() -> KlaxInitializer:
    def init(
        key: PRNGKeyArray,
        shape: Shape,
        fan_in: int,
        dtype: DTypeLikeInexact = jnp.float_,
    ) -> Array:
        mean = jnp.sqrt(
            0.5475114234402735 * fan_in / (1.3450928843923602 + fan_in)
        )
        return jnp.full(shape, mean, dtype=dtype)

    return init
