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

import typing
from typing import Any, Protocol, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar


@typing.runtime_checkable
class Loss(Protocol):
    """A callable loss object."""

    def __call__(
        self,
        model: PyTree,
        data: PyTree,
        batch_axis: int | None | Sequence[Any],
    ) -> Scalar:
        raise NotImplementedError


# def make_batched_xy_loss(loss_core: Callable[[Array, Array], Scalar]) -> Loss:
#    """Returns an object of type ``Loss`` for paired data of the form (x, y).
#
#    The retured function, first applies ``jax.vmap`` to the passed model. Then
#    `x` is passed to the mapped model to optain `y_pred`. Lastly `y_pred` and
#    `y` are passed to the ``loss_core`` function to compute the final loss.
#
#    Example:
#        The function may be used as a decorator in the following way:
#
#        >>> import jax.numpy as jnp
#        >>> from klax.losses import make_batched_xy_loss
#        >>>
#        >>> def model(x):
#        ...     return 2 * x
#        ...
#        >>> @make_batched_xy_loss
#        ... def loss(y_pred, y):
#        ...     return jnp.mean((y_pred - y)**2)
#        ...
#        >>> x = jnp.array([1., 1.])
#        >>> y = jnp.array([2., 2.])
#        >>> loss(model, (x, y))
#        Array(0., dtype=float32)
#
#    Args:
#        loss_core: The loss function taking the predicted output the ground
#            truth as two separate inputs.
#
#    Returns:
#        A callable ``Loss`` object that applies ``jax.vmap`` to a model and
#        computed the loss value using the ``loss_core`` function.
#    """
#
#    def loss(
#        model: PyTree, data: PyTree, batch_axis: int | None | Sequence[Any] = 0
#    ) -> Scalar:
#        x, y = data
#        if isinstance(batch_axis, tuple):
#            in_axes = batch_axis[0]
#        else:
#            in_axes = batch_axis
#        y_pred = jax.vmap(model, in_axes=(in_axes,))(x)
#        return loss_core(y_pred, y)
#
#    return loss


class MSE(Loss):
    """Mean squared error for data of shape (x, y)."""

    def __call__(
        self,
        model: PyTree,
        data: PyTree,
        batch_axis: int | None | Sequence[Any] = 0,
    ) -> Scalar:
        x, y = data
        if isinstance(batch_axis, tuple):
            in_axes = batch_axis[0]
        else:
            in_axes = batch_axis
        y_pred = jax.vmap(model, in_axes=(in_axes,))(x)
        return jnp.mean(jnp.square(y_pred - y))


mse = MSE()


class MAE(Loss):
    """Mean absolute error for data of shape (x, y)."""

    def __call__(
        self,
        model: PyTree,
        data: PyTree,
        batch_axis: int | None | Sequence[Any] = 0,
    ) -> Scalar:
        x, y = data
        if isinstance(batch_axis, tuple):
            in_axes = batch_axis[0]
        else:
            in_axes = batch_axis
        y_pred = jax.vmap(model, in_axes=(in_axes,))(x)
        return jnp.mean(jnp.abs(y_pred - y))


mae = MAE()
