import typing
from typing import Any, Protocol, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar

from ._typing import DataTree


@typing.runtime_checkable
class Loss(Protocol):
    def __call__(
        self,
        model: PyTree,
        x: DataTree,
        y: DataTree,
        in_axes: int | None | Sequence[Any]
    ) -> Scalar:
        raise NotImplementedError


def mse(
    model: PyTree,
    x: DataTree,
    y: DataTree,
    in_axes: int | None | Sequence[Any] = 0
) -> Scalar:
    y_pred = jax.vmap(model, in_axes=in_axes)(x)
    return jnp.mean((y_pred - y) ** 2)


def mae(
    model: PyTree,
    x: DataTree,
    y: DataTree,
    in_axes: int | None | Sequence[Any] = 0
) -> Scalar:
    y_pred = jax.vmap(model, in_axes=in_axes)(x)
    return jnp.mean(jnp.abs(y_pred - y))