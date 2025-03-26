import typing
from typing import Any, Callable, Protocol, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar, Array

from .typing import DataTree


@typing.runtime_checkable
class Loss(Protocol):
    def __call__(
        self,
        model: PyTree,
        data: DataTree,
        batch_axis: int | None | Sequence[Any]
    ) -> Scalar:
        raise NotImplementedError


def expects_tuple_and_vmap(loss: Callable[[Array, Array], Scalar]):
    """
    Wrapper for loss functions that splits the data into two,
    vmaps the model and then passes the output prediction and 
    the ground truth output to the wrapped function.

    Args:
        loss: Loss function taking the output prediction and
            ground truth as input.
    """
    def new_loss(
        model: PyTree,
        data: DataTree,
        batch_axis: int | None | Sequence[Any] = 0
    ) -> Scalar:
        x, y = data
        in_axes, _ = batch_axis
        y_pred = jax.vmap(model, in_axes=(in_axes,))(x)
        return loss(y_pred, y)
    return new_loss

@expects_tuple_and_vmap
def mse(y_pred, y):
    return jnp.mean(jnp.square(y_pred - y))

@expects_tuple_and_vmap
def mae(y_pred, y):
    return jnp.mean(jnp.abs(y_pred - y))