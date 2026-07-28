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
from functools import wraps
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Scalar

from ._wrappers import unwrap


class Loss(ABC):
    """An abstract callable loss object.

    Inherit from this class to define a custom loss that can be passed to
    [`fit`][klax.fit].
    An instance of the loss class has two methods - `value` and `value_and_grad` -
    which determine how the loss value and it's gradients are calculated.
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
        ...     def value(self, model, data, run_state):
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
        run_state: PyTree[Any],
    ) -> Scalar | tuple[Scalar, dict[str, Array]]:
        """Abstract method to compute the loss for a given model and data.

        Args:
            model: The model parameters or structure to evaluate the loss.
            batch: The input data or structure used for loss computation.
            run_state: Auxiliary, user-defined runtime state.

        Returns:
            Scalar or Tuple: If the output is a scalar, it is interpreted as
                the computed loss value. If it is a tuple, it must be a tuple
                of `Scalar` and `aux`, where the scalar is the
                computed loss value and `aux` is a dict of auxiliary quantities
                (e.g. loss components) to expose for logging/metrics.

        """
        pass

    def value_and_aux[T](
        self,
        model: PyTree,
        batch: PyTree[Any, "T"],
        run_state: PyTree[Any],
    ) -> tuple[Scalar, dict[str, Array]]:
        """Compute the loss value and aux.

        This method unwraps the model before computing the loss by calling
        the `value` method. It then normalizes the output of `value` by
        adding an empty dictionary as `aux` if `value` does not return any
        `aux`.

        Args:
            model: The model parameters or structure to evaluate the loss.
            batch: The input data or structure used for loss computation.
            run_state: Auxiliary, user-defined runtime state.

        Returns:
            Tuple: `Scalar` and `aux`, where the scalar is the
                computed loss value and `aux` is a dict of auxiliary quantities
                (e.g. loss components) to expose for logging/metrics.

        """
        model = unwrap(model)
        result = self.value(model, batch, run_state)
        if isinstance(result, tuple):
            loss, aux = result
            return loss, aux
        return result, {}

    def __call__[T](
        self,
        model: PyTree,
        batch: PyTree[Any, "T"],
        run_state: PyTree[Any],
    ) -> Scalar:
        """Compute the loss value used during training.

        This method unwraps the model before computing the loss by calling
        the `value` method.

        Args:
            model: The model parameters or structure to evaluate the loss.
            batch: The input data or structure used for loss computation.
            run_state: Auxiliary, user-defined runtime state.

        Returns:
            Scalar: The computed loss value.

        """
        loss, _ = self.value_and_aux(model, batch, run_state)
        return loss

    def value_and_grad[T, M](
        self,
        model: PyTree[Any, "M"],
        batch: PyTree[Any, "T"],
        run_state: PyTree[Any],
    ) -> tuple[tuple[Scalar, dict[str, Array]], PyTree[Any, "M"]]:
        """Compute the loss value and its gradient.

        This method computes the loss value and its gradient with respect to
        the model parameters by applying `eqx.filter_value_and_grad` to the `value`
        method.

        Args:
            model: The model parameters or structure to evaluate the loss.
            batch: The input data or structure used for loss computation.
            run_state: Auxiliary, user-defined runtime state.

        Returns:
            Tuple of loss value and gradient with respect to the model.

        """
        return eqx.filter_value_and_grad(self.value_and_aux, has_aux=True)(
            model, batch, run_state
        )

    def partitioned_value[T](
        self,
        params: PyTree,
        static: PyTree,
        batch: PyTree[Any, "T"],
        run_state: PyTree[Any],
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
            run_state: Auxiliary, user-defined runtime state.

        Returns:
            Scalar: The computed loss value.

        """
        model = eqx.combine(params, static)
        return self(model, batch, run_state)


def loss(
    func: Callable[
        [PyTree, PyTree, PyTree], Scalar | tuple[Scalar, dict[str, Array]]
    ],
) -> Loss:
    """Convert a function into a [`klax.Loss`][] object.

    Example:
        To create a mean squared error loss using this decorator, you can do:
        ```python
        @klax.loss
        def mse(model, data, run_state):
            x, y = data
            y_pred = jax.vmap(model)(x)
            return jnp.mean(jnp.square(y_pred - y))
        ```

    Args:
        func: Function that computes the loss. It must have the signature
            `(model: PyTree, batch: PyTree, run_state: PyTree) -> Scalar`.

    Returns:
        Loss: An instance of a subclass of [`klax.Loss`][] that wraps the given
            function.

    """

    class FuncLoss(Loss):
        @wraps(func)
        def value(self, model, batch, run_state):
            return func(model, batch, run_state)

    return FuncLoss()


@loss
def mse(model, data, run_state):
    """Mean squared error for a tuple of data `(x, y)`.

    The inputs `x` and the outputs `y` are expected to have the same batch axis
    and equal length along that axis.
    """
    x, y = data
    y_pred = jax.vmap(model)(x)
    return jnp.mean(jnp.square(y_pred - y))


@loss
def mae(model, data, run_state):
    """Mean absolute error for a tuple of data `(x, y)`.

    The inputs `x` and the outputs `y` are expected to have the same batch axis
    and equal length along that axis.
    """
    x, y = data
    y_pred = jax.vmap(model)(x)
    return jnp.mean(jnp.abs(y_pred - y))
