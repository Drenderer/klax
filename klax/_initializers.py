from collections.abc import Sequence
from typing import Any

from jax import numpy as jnp
from jax.nn.initializers import Initializer
from jaxtyping import Array

# Types from JAX
DTypeLikeInexact = Any
Shape = Sequence[int | Any]


def hoedt_normal(
    in_axis: int = -2,
    dtype: DTypeLikeInexact = jnp.float_,
) -> Initializer:
    def init(
        key: Array, shape: Shape, dtype: DTypeLikeInexact = dtype
    ) -> Array:
        fan_in = shape[in_axis]
        mean = ...

    return init
