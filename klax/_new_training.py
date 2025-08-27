from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import optax
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from ._wrappers import apply, unwrap


class DataHandler[T](ABC):
    train_data: PyTree[Any, "T"]
    validation_data: PyTree[Any, "T"] | None
    batch_axes: PyTree[int | None, "T ..."]  # type: ignore
    batch_size: int
    ...

    @abstractmethod
    def get_training_batch(
        self,
    ) -> PyTree[Any, "T"]:
        pass


class Loss(ABC):
    @abstractmethod
    def value[T](
        self,
        model: PyTree,
        batch: PyTree[Any, "T"],
        batch_axis: PyTree[int | None, "T ..."],  # type: ignore
    ) -> Scalar:
        pass

    def value_and_grad[T, M](
        self,
        model: PyTree[Any, "M"],
        batch: PyTree[Any, "T"],
        batch_axis: PyTree[int | None, "T ..."],  # type: ignore
    ) -> tuple[Scalar, PyTree[Any, "M"]]:
        return jax.value_and_grad(self.value)(model, batch, batch_axis)


class DefaultLoss(Loss):
    loss_fn: Callable

    def value(self, model, batch, batch_axis):
        model = unwrap(model)
        return self.loss_fn(model, batch, batch_axis=batch_axis)


@dataclass
class TrainingState:
    # Replaces CallbackArgs -> Enables modifying every training aspect through callbacks
    model: PyTree
    datahandler: DataHandler
    optimizer: optax.GradientTransformation
    optimizer_state: PyTree
    loss: Loss
    step: int
    steps: int


class Callback(ABC):
    """An abstract callback.

    Inherit from this class to create a custom callback.
    """

    def __call__(self, training_state: TrainingState) -> bool | None:
        """Call after each step during training."""
        pass

    def on_training_end(self, training_state: TrainingState) -> None:
        """Call when training ends."""
        pass

    def on_training_start(self, training_state: TrainingState) -> None:
        """Call when training starts."""
        pass


def training_loop(
    training_state: TrainingState, callbacks: Iterable[Callback] = []
):
    @eqx.filter_jit
    def make_step(batch, model, optimizer, optimizer_state):
        # Where can this function go? Seems wrong to put it here
        # Can we make it a method of training state without interfering with jit?
        value, grad = training_state.loss.value_and_grad(
            model, batch, training_state.datahandler.batch_axes
        )
        updates, optimizer_state = optimizer.update(
            grad, optimizer_state, value=value
        )
        model = optax.apply_updates(model, updates)
        model = apply(model)
        return model, optimizer_state

    for callback in callbacks:
        callback.on_training_start(training_state)

    for training_state.step in range(1, training_state.steps + 1):
        batch = training_state.datahandler.get_training_batch()
        training_state.model, training_state.optimizer_state = make_step(
            batch,
            training_state.model,
            training_state.optimizer,
            training_state.optimizer_state,
        )
        if any([callback(training_state) for callback in callbacks]):
            break

    for callback in callbacks:
        callback.on_training_end(training_state)

    return training_state


def fit(model, data, validation_data, loss_fn):
    # Initialize training state and callbacks
    loss = DefaultLoss(loss_fn)
    training_state = TrainingState(model, loss=loss)
    callbacks.append(history)
    training_state = training_loop(training_state, callbacks)
    return training_state.model, history
