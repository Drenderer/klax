import typing
from typing import Any, Callable, Protocol, Sequence, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar, Array

from .typing import DataTree


@typing.runtime_checkable
class Loss(Protocol):
    """A callable loss object."""
    def __call__(
        self, model: PyTree, data: DataTree, batch_axis: int | None | Sequence[Any]
    ) -> Scalar:
        raise NotImplementedError


def make_batched_xy_loss(loss_core: Callable[[Array, Array], Scalar]) -> Loss:
    """Returns an object of type ``Loss`` for paired data of the form (x, y).

    The retured function, first applies ``jax.vmap`` to the passed model. Then
    `x` is passed to the mapped model to optain `y_pred`. Lastly `y_pred` and
    `y` are passed to the ``loss_core`` function to compute the final loss.

    Example:
        The function may be used as a decorator in the following way:

        >>> import jax.numpy as jnp
        >>> from klax.losses import make_batched_xy_loss
        >>>
        >>> def model(x):
        ...     return 2 * x
        ...
        >>> @make_batched_xy_loss
        ... def loss(y_pred, y):
        ...     return jnp.mean((y_pred - y)**2)
        ...
        >>> x = jnp.array([1., 1.])
        >>> y = jnp.array([2., 2.])
        >>> loss(model, (x, y))
        Array(0., dtype=float32)

    Args:
        loss_core: The loss function taking the predicted output the ground
            truth as two separate inputs.

    Returns:
        A callable ``Loss`` object that applies ``jax.vmap`` to a model and
        computed the loss value using the ``loss_core`` function.
    """

    def loss(
        model: PyTree, data: DataTree, batch_axis: int | None | Sequence[Any] = 0
    ) -> Scalar:
        x, y = data
        if isinstance(batch_axis, tuple):
            in_axes = batch_axis[0]
        else:
            in_axes = batch_axis
        y_pred = jax.vmap(model, in_axes=(in_axes,))(x)
        return loss_core(y_pred, y)

    return loss


@make_batched_xy_loss
def mse(y_pred, y):
    return jnp.mean(jnp.square(y_pred - y))


@make_batched_xy_loss
def mae(y_pred, y):
    return jnp.mean(jnp.abs(y_pred - y))
