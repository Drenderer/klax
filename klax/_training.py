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
from typing import Any

import equinox as eqx
import jax
import optax
from jaxtyping import PRNGKeyArray, PyTree

from klax._losses import Loss

from ._callbacks import Callback
from ._datahandler import (
    BatchGenerator,
    batch_data,
)
from ._logging import History, LossMetric, MetricLogger
from ._losses import Loss, mse
from ._trainstate import (
    TrainingState,
    TrainingStatic,
    TrainingView,
    make_state_and_static,
)
from ._wrappers import apply

new_logger = object()


@eqx.filter_jit
def make_step(
    state: TrainingState, batch: PyTree[Any], static: TrainingStatic
) -> TrainingState:
    # Assembling the model here provides a clear separation between
    # static and dynamic parts of the training loop.
    # Furthermore it implements the unflattening trick described in
    # [low-overhead training loops][https://docs.kidger.site/equinox/tricks/].
    # This slightly reduces JAX's overhead when repeatedly passing through the
    # jit boundary of the make_step function in the training loop.
    model = static.assemble_model(state.model_leaves)
    opt_state = static.assemble_opt_state(state.opt_state_leaves)

    model_params, model_static = eqx.partition(model, eqx.is_inexact_array)
    value, grad = static.loss.value_and_grad(model, batch, static.batch_axes)
    updates, opt_state = static.optimizer.update(
        grad,
        opt_state,
        model_params,
        value=value,
        grad=grad,
        value_fn=jax.tree_util.Partial(
            static.loss.partitioned_value,
            static=model_static,
            batch=batch,
            batch_axes=static.batch_axes,
        ),
    )
    model_params = optax.apply_updates(model_params, updates)
    model = eqx.combine(model_params, model_static)

    # Apply the constraints to ensure they are met again after the update.
    model = apply(model)

    new_state = TrainingState(
        model_leaves=static.disassemble_model(model),
        opt_state_leaves=static.disassemble_opt_state(opt_state),
    )
    return new_state


def run_training_loop(
    state: TrainingState,
    static: TrainingStatic,
    callbacks: Iterable[Callback],
):
    step = 0
    view = TrainingView(state, static)
    for callback in callbacks:
        callback.on_training_start(view, step)

    for step in range(1, static.steps + 1):
        state = make_step(state, next(static.batch), static)

        view = TrainingView(state, static)
        stop = False
        for callback in callbacks:
            stop |= bool(callback(view, step))
        if stop:
            break

    for callback in callbacks:
        callback.on_training_end(view, step)

    return state


def fit[T: eqx.Module, H: Callback](
    model: T,
    data: PyTree[Any],
    *,
    batch_size: int = 32,
    batch_axes: PyTree[int | None] = 0,
    validation_data: PyTree[Any] = None,
    steps: int = 1000,
    loss: Loss = mse,
    optimizer: optax.GradientTransformation
    | optax.GradientTransformationExtraArgs = optax.adam(1e-3),
    init_opt_state: PyTree[Any] = None,
    batcher: BatchGenerator = batch_data,
    logger: MetricLogger | None = new_logger,
    callbacks: Iterable[Callback] | None = None,
    key: PRNGKeyArray,
) -> tuple[T, History]:
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
        logger: A callback intended for tracking the training process. If no
            custom callback is passed the [`klax.MetricLogger`][] with a
            logging interval of 100 steps is used. To change the logging
            increment or verbosity of this default callback, pass a
            `MetricLogger` object to this argument, e.g.,
            `logger=MetricLogger(log_every=10, verbose=False)` for logging
            on every 10-th step without printing the loss. This can also be used
            to log additional metrics during training, e.g.,
            ```python
                mylogger=MetricLogger(log_every=100)
                mylogger.add_metric(
                    "accuracy",
                    lambda model: compute_accuracy(model)
                )
                model, history = fit(
                    model,
                    data,
                    ...,
                    logger=mylogger,
                )
            ```
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

    bkey, key = jax.random.split(key)
    batch = batcher(data, batch_size, batch_axes, key=bkey)
    state, static = make_state_and_static(
        model,
        optimizer,
        opt_state,
        batch,
        batch_axes,
        loss,
        steps,
    )

    # Make callbacks iterable
    callbacks = [] if callbacks is None else list(callbacks)

    # Initialize callback arguments and history
    if logger is not None:
        logger = MetricLogger() if logger is new_logger else logger

        bkey, key = jax.random.split(key)
        logger.add_metric(
            "loss",
            LossMetric(batcher, data, batch_size, batch_axes, loss, key=bkey),
            verbose=True,
        )

        if validation_data is not None:
            bkey, key = jax.random.split(key)
            logger.add_metric(
                "validation_loss",
                LossMetric(
                    batcher,
                    validation_data,
                    4 * batch_size,
                    batch_axes,
                    loss,
                    key=bkey,
                ),
                verbose=True,
            )
        callbacks.append(logger)

    state = run_training_loop(state, static, callbacks)

    model = static.assemble_model(state.model_leaves)

    history = logger.history if logger is not None else History()

    return model, history
