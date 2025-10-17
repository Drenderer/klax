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
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar


@typing.runtime_checkable
class Loss(Protocol):
    """An abstract callable loss object.

    It can be used to build custom losses that can be passed to [`klax.fit`][].

    Example:
        A simple custom loss that computes the mean squared error between
        the predicted values `y_pred` and true values `y` for in inputs `x` may
        be implemented as follows:

        ```python
        >>> def mse(model, data, batch_axes=0):
        ...    x, y = data
        ...    if isinstance(batch_axes, tuple):
        ...        in_axes = batch_axes[0]
        ...    else:
        ...        in_axes = batch_axes
        ...    y_pred = jax.vmap(model, in_axes=(in_axes,))(x)
        ...    return jnp.mean(jnp.square(y_pred - y))
        ```

        Note that, since we a aim to provide a maximum of flexibility the users
        have to take care of applying `jax.vmap` to the model themselves.

    """

    @abstractmethod
    def __call__(
        self,
        model: PyTree,
        batch: PyTree,
        batch_axes: int | None | Sequence[Any],
    ) -> Scalar:
        """Abstract method to compute the loss for a given model and data.

        Args:
            model: The model parameters or structure to evaluate the loss.
            batch: A batch of input data used for loss computation.
            batch_axes: Specifies the axis or axes corresponding to the batch
                dimension in the data. Can be an integer, None, or a sequence
                of values.

        Returns:
            Scalar: The computed loss value.

        """
        ...


class MSE(Loss):
    """Mean squared error for a batch of data of the form `(x, y)`.

    The inputs `x` and the outputs `y` are expected to have the same batch axis
    and equal length along that axis.
    """

    def __call__(
        self,
        model: PyTree,
        batch: PyTree,
        batch_axes: int | None | Sequence[Any] = 0,
    ) -> Scalar:
        x, y = batch
        if isinstance(batch_axes, tuple):
            in_axes = batch_axes[0]
        else:
            in_axes = batch_axes
        y_pred = jax.vmap(model, in_axes=(in_axes,))(x)
        return jnp.mean(jnp.square(y_pred - y))


mse = MSE()


class MAE(Loss):
    """Mean absolute error for a batch of data of the form `(x, y)`.

    The inputs `x` and the outputs `y` are expected to have the same batch axis
    and equal length along that axis.
    """

    def __call__(
        self,
        model: PyTree,
        batch: PyTree,
        batch_axes: int | None | Sequence[Any] = 0,
    ) -> Scalar:
        x, y = batch
        if isinstance(batch_axes, tuple):
            in_axes = batch_axes[0]
        else:
            in_axes = batch_axes
        y_pred = jax.vmap(model, in_axes=(in_axes,))(x)
        return jnp.mean(jnp.abs(y_pred - y))


mae = MAE()
