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

"""Implements a basic training loop."""

from collections.abc import Iterable
from functools import partial
from typing import Any, overload

import equinox as eqx
import optax
from jaxtyping import PRNGKeyArray, PyTree

from ._callbacks import (
    Callback,
    HistoryCallback,
    TrainingState,
    TrainingStatic,
)
from ._datahandler import (
    BatchGenerator,
    batch_data,
)
from ._losses import Loss, mse
from ._trainstate import TrainingState, TrainingStatic
from ._wrappers import apply


def run_training_loop(
    state: TrainingState,
    static: TrainingStatic,
    callbacks: Iterable[Callback] = [],
):
    @eqx.filter_jit
    def make_step(state, batch):
        value, grad = static.loss.value_and_grad(
            state.model, batch, static.batch_axes
        )
        updates, opt_state = static.optimizer.update(
            grad,
            state.opt_state,
            value=value,
            grad=grad,
        )
        model = apply(eqx.apply_updates(state.model, updates))
        state.model = model
        state.opt_state = opt_state
        return state, value

    for callback in callbacks:
        callback.on_training_start(state, static)

    for step in range(1, static.steps + 1):
        state, batch_loss = make_step(state, next(static.batcher))

        # Run all callbacks and break if any of them request termination of
        # the training loop.
        # Note! The square brackets are important. Otherwise the loop is
        # terminated with the first callback that returns true. But we want
        # to run all callbacks first and then decide, whether to terminate.
        if any(
            [
                callback(state, step, batch_loss, static)
                for callback in callbacks
            ]
        ):
            break

    for callback in callbacks:
        callback.on_training_end(state, static)

    return state


@overload
def fit[T: eqx.Module](
    model: T,
    data: PyTree[Any],
    *,
    batch_size: int = 32,
    batch_axes: PyTree[int | None] = 0,
    validation_data: PyTree[Any] = None,
    steps: int = 1000,
    loss: Loss = mse,
    optimizer: optax.GradientTransformationExtraArgs = optax.adam(1e-3),
    init_opt_state: PyTree[Any] = None,
    batcher: BatchGenerator = batch_data,
    history: None = None,
    callbacks: Iterable[Callback] | None = None,
    key: PRNGKeyArray,
) -> tuple[T, HistoryCallback]: ...
@overload
def fit[T: eqx.Module, H: Callback](
    model: T,
    data: PyTree[Any],
    *,
    batch_size: int = 32,
    batch_axes: PyTree[int | None] = 0,
    validation_data: PyTree[Any] = None,
    steps: int = 1000,
    loss: Loss = mse,
    optimizer: optax.GradientTransformationExtraArgs = optax.adam(1e-3),
    init_opt_state: PyTree[Any] = None,
    batcher: BatchGenerator = batch_data,
    history: H,
    callbacks: Iterable[Callback] | None = None,
    key: PRNGKeyArray,
) -> tuple[T, H]: ...
def fit[T: eqx.Module, H: Callback](
    model: T,
    data: PyTree[Any],
    *,
    batch_size: int = 32,
    batch_axes: PyTree[int | None] = 0,
    validation_data: PyTree[Any] = None,
    steps: int = 1000,
    loss: Loss = mse,
    optimizer: optax.GradientTransformationExtraArgs = optax.adam(1e-3),
    init_opt_state: PyTree[Any] = None,
    batcher: BatchGenerator = batch_data,
    history: HistoryCallback | H | None = None,
    callbacks: Iterable[Callback] | None = None,
    key: PRNGKeyArray,
) -> tuple[T, HistoryCallback | H]:
    """Trains a model using an optimizer from optax.

    This is a convenient wrapper around `training_loop` which sets up optimizer,
    training state and callbacks.

    Args:
        model: The model instance, which should be trained. It must be a
            subclass of `equinox.Module`. The model may contain
            [`klax.Unwrappable`][] wrappers.
        data: The training data can be any `PyTree` with `ArrayLike` leaves.
            Most likely you'll want `data` to be a tuple `(x, y)` with model
            inputs `x` and model outputs `y`.
        batch_size: The number of examples in a batch.
        batch_axes: A `PyTree` denoting, which axis is the batch axis for
            arrays in `data`. `batch_axes` must be a prefix of `data`. By
            specifying `batch_axes` as a `PyTree` it is possible to specify
            different batch axes for different leaves of `data`. (Defaults to
            `0`, meaning the first axes of arrays in `data` are batch
            dimensions.)
        validation_data: Arbitrary `PyTree` used for validation during
            training. Must have the same tree structure as `data`. (Defaults
            to None.)
        steps: Number of gradient updates to apply. (Defaults to 1000.)
        loss: The loss function with call signature
            `(model: PyTree, data: PyTree, batch_axes: int | None |
            Sequence[Any]) -> float`. (Defaults to `mse`.)
        optimizer: The optimizer. Any optax gradient transform to calculate
            the updates for the model. (Defaults to optax.adam(1e-3).)
        init_opt_state: The initial state of the optimizer. If `None`, the
            optimizer is initialized from scratch. By providing a value for
            `init_opt_state`, the user can resume training from a previous
            state (e.g., obtained from the `HistoryCallback.last_opt_state`).
            (Defaults to `None`.)
        batcher: The data loader that splits inputs and targets into batches.
            (Defaults to `batch_data`.)
        history: A callback intended for tracking the training process. If no
            custom callback is passed the [`klax.HistoryCallback`][] with a
            logging interval of 100 steps is used. To change the logging
            increment or verbosity of this default callback, pass a
            `HistoryCallback` object to this argument, e.g.,
            `history=HistoryCallback(log_every=10, verbose=False)` for logging
            on every 10-th step without printing the loss.
        callbacks: Callback functions that are evaluated after every training
            step. They can be used to implement early stopping, custom history
            logging and more. The argument to the callback function is a
            CallbackArgs object. (Defaults to `None`. Keyword only Argument)
        key: A `jax.random.PRNGKey` used to provide randomness for batch
            generation. (Keyword only argument.)

    Note:
        This function assumes that the batch dimension is always oriented along
        the first axes of any `jax.Array`

    Returns:
        A tuple of the trained model and the loss history.

    """
    if init_opt_state is None:
        # Initialize the optimizer and 'tell it' to optimize with respect to
        # all inexact arrays in the model. This is done by passing the model to
        # the optimizer.
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    else:
        opt_state = init_opt_state

    # Apply the Constraint in the model to ensure apply-constrains are met
    # initially
    model = apply(model)

    state = TrainingState(model=model, opt_state=opt_state)
    static = TrainingStatic(
        optimizer=optimizer,
        batcher=batcher(
            data,
            batch_size,
            batch_axes,
            key=key,
        ),
        batch_axes=batch_axes,
        loss=loss,
        steps=steps,
    )

    # Make callbacks iterable
    callbacks = [] if callbacks is None else list(callbacks)

    # Initialize callback arguments and history
    if history is None:
        metric_defs = {
            "training_loss": partial(
                loss.value, batch=data, batch_axes=batch_axes
            )
        }
        if validation_data is not None:
            metric_defs["validation_loss"] = partial(
                loss.value, batch=validation_data, batch_axes=batch_axes
            )
        history = HistoryCallback(
            metric_defs=metric_defs,
            log_every=100,
        )
    callbacks.append(history)

    state = run_training_loop(state, static, callbacks)

    return state.model, history
