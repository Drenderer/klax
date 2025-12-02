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

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar

from ._wrappers import unwrap


class Loss(ABC):
    """An abstract callable loss object.

    It can be used to build custom losses that can be passed to [`klax.fit`][].
    An instance of the loss class has two methods that are required for
    [`klax.fit`][`fit`]: `value` and `value_and_grad`. In most cases the default
    implementation should be used. `value` just [`klax.unwrap`][unwraps] the model
    before computing the loss as specified in `__call__`, while `value_and_grad`
    per default applies `jax.value_and_grad` to `value`. These functions can be
    overwritten, for example to enable custom calculations of the gradients.

    Example:
        A simple custom loss that computes the mean squared error between
        the predicted values `y_pred` and true values `y` for in inputs `x` may
        be implemented as follows:

        ```python
        >>> class MSE(klax.Loss):
        ...     def __call__(self, model, data, batch_axes):
        ...         x, y = data
        ...         y_pred = jax.vmap(model, in_axes=batch_axes)(x)
        ...         return jnp.mean(jnp.square(y_pred - y))
        ```

        Note that, since we a aim to provide a maximum of flexibility the users
        have to take care of applying `jax.vmap` to the model themselves.

    """

    @abstractmethod
    def __call__[T](
        self,
        model: PyTree,
        batch: PyTree[Any, "T"],
        batch_axes: PyTree[int | None, "T ..."],  # type: ignore
    ) -> Scalar:
        """Abstract method to compute the loss for a given model and data.

        Args:
            model: The model parameters or structure to evaluate the loss.
            batch: The input data or structure used for loss computation.
            batch_axes: Specifies the axis or axes corresponding to the batch
                dimension in the data. Can be an integer, None, or a sequence
                of values.

        Returns:
            Scalar: The computed loss value.

        """
        pass

    def value[T](
        self,
        model: PyTree,
        batch: PyTree[Any, "T"],
        batch_axes: PyTree[int | None, "T ..."],  # type: ignore
    ) -> Scalar:
        """Compute the loss value during training.

        Args:
            model: The model parameters or structure to evaluate the loss.
            batch: The input data or structure used for loss computation.
            batch_axes: Specifies the axis or axes corresponding to the batch
                dimension in the data. Can be an integer, None, or a sequence
                of values.

        Returns:
            Scalar: The computed loss value.

        """
        model = unwrap(model)
        return self(model, batch, batch_axes)

    def value_and_grad[T, M](
        self,
        model: PyTree[Any, "M"],
        batch: PyTree[Any, "T"],
        batch_axes: PyTree[int | None, "T ..."],  # type: ignore
    ) -> tuple[Scalar, PyTree[Any, "M"]]:
        """Compute the loss value and gradient during training.

        Args:
            model: The model parameters or structure to evaluate the loss.
            batch: The input data or structure used for loss computation.
            batch_axes: Specifies the axis or axes corresponding to the batch
                dimension in the data. Can be an integer, None, or a sequence
                of values.

        Returns:
            Tuple of loss value and gradient with respect to the model.

        """
        return eqx.filter_value_and_grad(self.value)(model, batch, batch_axes)


def loss(func: Callable):
    """Convert a function into a [`klax.Loss`][] object.

    Args:
        func: Function that computes the loss. It must have the signature
            `(model: PyTree, batch: PyTree, batch_axes: PyTree) -> Scalar`.

    Returns:
        Loss: An instance of a subclass of [`klax.Loss`][] that wraps the given
            function.

    """

    class FuncLoss(Loss):
        def __call__(self, model, batch, batch_axes):
            return func(model, batch, batch_axes)

    return FuncLoss()


@loss
def mse(model, data, batch_axes):
    """Mean squared error for a tuple of data `(x, y)`.

    The inputs `x` and the outputs `y` are expected to have the same batch axis
    and equal length along that axis.
    """
    x, y = data
    y_pred = jax.vmap(model, in_axes=batch_axes)(x)
    return jnp.mean(jnp.square(y_pred - y))


@loss
def mae(model, data, batch_axes):
    """Mean absolute error for a tuple of data `(x, y)`.

    The inputs `x` and the outputs `y` are expected to have the same batch axis
    and equal length along that axis.
    """
    x, y = data
    y_pred = jax.vmap(model, in_axes=batch_axes)(x)
    return jnp.mean(jnp.abs(y_pred - y))
