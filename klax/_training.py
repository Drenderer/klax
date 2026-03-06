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

from collections.abc import Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import optax
from jaxtyping import PRNGKeyArray, PyTree, PyTreeDef

from klax._losses import Loss

from ._callbacks import Callback
from ._datahandler import (
    Batcher,
    batch_data,
)
from ._logging import BatchMetric, History, Metric, MetricLogger
from ._losses import Loss, mse
from ._trainstate import TrainingContext, TrainingState
from ._wrappers import apply

type Leaf = Any


@eqx.filter_jit
def make_step(
    state_leaves: list[Leaf],
    state_treedef: Any,
    batch: PyTree,
    loss: Loss,
    optimizer: optax.GradientTransformationExtraArgs,
) -> tuple[list[Leaf], Any]:
    state = jax.tree.unflatten(state_treedef, state_leaves)

    model_params, model_static = eqx.partition(
        state.model, eqx.is_inexact_array
    )
    value, grad = loss.value_and_grad(state.model, batch, state.run_state)
    updates, opt_state = optimizer.update(
        grad,
        state.opt_state,
        model_params,
        value=value,
        grad=grad,
        value_fn=jax.tree_util.Partial(
            loss.partitioned_value,
            static=model_static,
            batch=batch,
            run_state=state.run_state,
        ),
    )
    model_params = optax.apply_updates(model_params, updates)
    model = eqx.combine(model_params, model_static)

    # Apply the constraints to ensure they are met again after the update.
    model = apply(model)

    step = state.step + 1

    state = TrainingState(model, opt_state, state.run_state, step)

    return jax.tree.flatten(state)


def run_training_loop(
    context: TrainingContext,
    callbacks: Sequence[Callback],
) -> TrainingContext:
    state_leaves = context._state_leaves
    state_treedef = context._state_treedef

    for callback in callbacks:
        callback.on_training_start(context)

    for batch in context.batch_generator:
        if context.state.step >= context.steps:
            break

        state_leaves, _ = make_step(
            state_leaves,
            state_treedef,
            batch,
            context.loss,
            context.optimizer,
        )
        context.update_state(state_leaves)

        stop = False
        for callback in callbacks:
            stop |= bool(callback.on_training_step(context))
        if stop:
            break

    for callback in callbacks:
        callback.on_training_end(context)

    return context


def fit[T: eqx.Module](
    model: T,
    data: PyTree[Any],
    *,
    batch_size: int = 32,
    batch_axes: PyTree[int | None] = 0,
    run_state: PyTree[Any] = None,
    validation_data: PyTree[Any] = None,
    steps: int = 1000,
    loss: Loss = mse,
    optimizer: optax.GradientTransformation
    | optax.GradientTransformationExtraArgs = optax.adam(1e-3),
    init_opt_state: PyTree[Any] = None,
    batcher: Batcher = batch_data,
    metrics: Sequence[Metric] | None = None,
    log_every: int = 100,
    verbose: Literal[0, 1, 2] = 2,
    callbacks: Sequence[Callback] | None = None,
    key: PRNGKeyArray,
) -> tuple[T, History]:
    """Train a model using an optimizer from optax.

    This is a convenient wrapper around [`run_training_loop`][klax.run_training_loop]
    that sets up optimizer, training state and callbacks.

    Args:
        model: The model instance, which should be trained. It must be a
            subclass of `equinox.Module`. The model may contain
            [`klax.Unwrappable`][] wrappers.
        data: The training data can be any `PyTree` with at least some
            `ArrayLike` leaves. Most likely you'll want `data` to be a
            tuple `(x, y)` with model inputs `x` and model outputs `y`.
        batch_size: The number of examples in a batch.
        batch_axes: A `PyTree` denoting, which axis is the batch axis for
            arrays in `data`. `batch_axes` must be a prefix of `data`. By
            specifying `batch_axes` as a `PyTree` it is possible to specify
            different batch axes for different leaves of `data`. (Defaults to
            `0`, meaning the first axes of arrays in `data` are batch
            dimensions.)

            Example: For a dataset of 100 examples `data = (x, (y1, y2), "some_string")`
            where `x` has  shape `(100, 32)`, `y1` has shape `(100,)`
            and `y2` has shape `(10, 100)`, the appropriate `batch_axes`
            would be `batch_axes = (0, (0, 1, None))` indicating that
            the batch axis for `x` is the first axis (0), for `y1` also
            the first axis (0), for `y2` the second axis (1) and for the
            string there is no batch axis (`None`).
            Defaults to `0`.
        run_state: Auxiliary runtime state, that is passed to the loss function.
            Can be updated via callbacks.
            Defaults to `None`.
        validation_data: Arbitrary `PyTree` used for validation during
            training. Must have the same tree structure as `data`. (Defaults
            to None.)
            Internally, the validation data is used to create a [BatchMetric][klax.BatchMetric]
            for logging. Each time the metric is evaluated, the loss is computed
            on a batch from the validation dataset with batch size `4*batch_size`.
            Defaults to `None`
        steps: Number of gradient updates to apply.
            Defaults to 1000.
        loss: The loss function with call signature
            `(model: PyTree, data: PyTree, run_state: PyTree) -> float`.
            Defaults to `mse`.
        optimizer: The optimizer. Any optax gradient transform to calculate
            the updates for the model.
            Defaults to optax.adam(1e-3).
        init_opt_state: The initial state of the optimizer. If `None`, the
            optimizer is initialized from scratch. By providing a value for
            `init_opt_state`, the user can resume training from a previous
            state (e.g., obtained from the `HistoryCallback.last_opt_state`).
            Defaults to `None`.
        batcher: The data loader that splits inputs and targets into batches.
            Defaults to `batch_data`.
        metrics: Sequence of [metrics][klax.Metric] to be evaluated at regular
            intervals during the training. You can overwrite the default "loss"
            and "validation_loss" metrics, by adding custom metrics with the same
            name.
            Defaults to `None`.
        log_every: Interval for both metric evaluation and progress logging.
            A value `log_every=n` means that every `n` steps during the training
            the metrics are evaluated and (if `verbose>0`) the training progress
            and selected metric values are printed.
        verbose: Integer controlling the verbosity during training.
            - 0: Nothing is printed.
            - 1: A message is printed every `log_every` steps.
            - 2: A progressbar is used and updated every `log_every` steps.
            Defaults to `2`.
        callbacks: List of [Callbacks][klax.Callback]. They can be used to
            implement early stopping, custom logging and more. The argument
            to the callback function is aCallbackArgs object.
            Defaults to `None`.
        key: A `jax.random.PRNGKey` used to provide randomness for batch
            generation.

    Returns:
        A tuple of the trained model and the training history.

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
    batch_generator = batcher(data, batch_size, batch_axes, key=bkey)
    context = TrainingContext(
        model, optimizer, opt_state, batch_generator, run_state, loss, steps
    )

    # Make callbacks iterable
    callbacks = [] if callbacks is None else list(callbacks)

    # Initialize logging and default metrics
    _metrics = []
    _metrics.append(
        BatchMetric(
            "loss",
            loss,
            data,
            batcher,
            batch_size,
            batch_axes,
            verbose=True,
            key=bkey,
        )
    )
    if validation_data is not None:
        _metrics.append(
            BatchMetric(
                "validation_loss",
                loss,
                validation_data,
                batcher,
                4 * batch_size,
                batch_axes,
                verbose=True,
                key=bkey,
            ),
        )
    if metrics is not None:
        _metrics += metrics
    logger = MetricLogger(log_every, _metrics, verbose)

    callbacks.append(logger)

    context = run_training_loop(context, callbacks)

    model = context.state.model

    history = logger.history if logger is not None else History()

    return model, history
