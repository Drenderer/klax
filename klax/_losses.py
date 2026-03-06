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
from functools import update_wrapper
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar

from ._wrappers import unwrap


class Loss(ABC):
    """An abstract callable loss object.

    Inherit from this class to define a custom loss that can be passed to
    [`fit`][klax.fit].
    An instance of the loss class has two methods that are required for
    [`fit`][klax.fit]: `value` and `value_and_grad`, which determine how
    the loss value and it's gradients are calculated.
    To define a custom loss, implement a custom `value` method. When calling
    the loss instance, the model will first be [unwrapped][klax.unwrap], and
    then passed to the `value` method.
    The `value_and_grad` function per default computes the gradient based on
    the `value` function. You should only overwrite it to specify a custom
    gradient computation.

    Example:
        A simple custom loss that computes the mean squared error between
        the predicted values `y_pred` and true values `y` for inputs `x` may
        be implemented as follows:

        ```python
        >>> class MSE(klax.Loss):
        ...     def value(self, model, data, aux):
        ...         x, y = data
        ...         y_pred = jax.vmap(model)(x)
        ...         return jnp.mean(jnp.square(y_pred - y))
        ```

    """

    @abstractmethod
    def value[T](
        self,
        model: PyTree,
        batch: PyTree[Any, "T"],
        aux: PyTree[Any],
    ) -> Scalar:
        """Abstract method to compute the loss for a given model and data.

        Args:
            model: The model parameters or structure to evaluate the loss.
            batch: The input data or structure used for loss computation.
            aux: Auxiliary, user-defined state.

        Returns:
            Scalar: The computed loss value.

        """
        pass

    def __call__[T](
        self,
        model: PyTree,
        batch: PyTree[Any, "T"],
        aux: PyTree[Any],
    ) -> Scalar:
        """Compute the loss value used during training.

        This method unwraps the model before computing the loss by calling
        the `value` method.

        Args:
            model: The model parameters or structure to evaluate the loss.
            batch: The input data or structure used for loss computation.
            aux: Auxiliary, user-defined state.

        Returns:
            Scalar: The computed loss value.

        """
        model = unwrap(model)
        return self.value(model, batch, aux)

    def value_and_grad[T, M](
        self,
        model: PyTree[Any, "M"],
        batch: PyTree[Any, "T"],
        aux: PyTree[Any],
    ) -> tuple[Scalar, PyTree[Any, "M"]]:
        """Compute the loss value and its gradient.

        This method computes the loss value and its gradient with respect to
        the model parameters by applying `eqx.filter_value_and_grad` to the `value`
        method.

        Args:
            model: The model parameters or structure to evaluate the loss.
            batch: The input data or structure used for loss computation.
            aux: Auxiliary, user-defined state.

        Returns:
            Tuple of loss value and gradient with respect to the model.

        """
        return eqx.filter_value_and_grad(self)(model, batch, aux)

    def partitioned_value[T](
        self,
        params: PyTree,
        static: PyTree,
        batch: PyTree[Any, "T"],
        aux: PyTree[Any],
    ) -> Scalar:
        """Compute the loss value for partitioned models.

        This method is useful when working with models that have been
        partitioned using Equinox's `partition` functionality. It combines
        the model from its parameter and static parts before computing the
        loss.

        Args:
            params: The parameter part of the partitioned model.
            static: The static part of the partitioned model.
            batch: The input data or structure used for loss computation.
            aux: Auxiliary, user-defined state.

        Returns:
            Scalar: The computed loss value.

        """
        model = eqx.combine(params, static)
        return self(model, batch, aux)


def loss(func: Callable[[PyTree, PyTree, PyTree], Scalar]) -> Loss:
    """Convert a function into a [`klax.Loss`][] object.

    Example:
        To create a mean squared error loss using this decorator, you can do:
        ```python
        @loss
        def mse(model, data, aux):
            x, y = data
            y_pred = jax.vmap(model)(x)
            return jnp.mean(jnp.square(y_pred - y))
        ```

    Args:
        func: Function that computes the loss. It must have the signature
            `(model: PyTree, batch: PyTree, aux: PyTree) -> Scalar`.

    Returns:
        Loss: An instance of a subclass of [`klax.Loss`][] that wraps the given
            function.

    """

    class FuncLoss(Loss):
        def value(self, model, batch, aux):
            return func(model, batch, aux)

    return update_wrapper(FuncLoss(), func)


@loss
def mse(model, data, aux):
    """Mean squared error for a tuple of data `(x, y)`.

    The inputs `x` and the outputs `y` are expected to have the same batch axis
    and equal length along that axis.
    """
    x, y = data
    y_pred = jax.vmap(model)(x)
    return jnp.mean(jnp.square(y_pred - y))


@loss
def mae(model, data, aux):
    """Mean absolute error for a tuple of data `(x, y)`.

    The inputs `x` and the outputs `y` are expected to have the same batch axis
    and equal length along that axis.
    """
    x, y = data
    y_pred = jax.vmap(model)(x)
    return jnp.mean(jnp.abs(y_pred - y))
