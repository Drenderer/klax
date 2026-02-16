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
from jaxtyping import PRNGKeyArray, PyTree

from klax._losses import Loss

from ._callbacks import Callback
from ._datahandler import (
    BatchGenerator,
    batch_data,
)
from ._logging import BatchMetric, History, Metric, MetricLogger
from ._losses import Loss, mse
from ._trainstate import (
    TrainingState,
    TrainingStatic,
    TrainingView,
    make_view,
)
from ._wrappers import apply

new_logger = object()


@eqx.filter_jit
def make_step(
    state: TrainingState, batch: PyTree, static: TrainingStatic
) -> TrainingState:
    """Update the training state by one optimization step.

    This function implements the unflattening trick described in
    [low-overhead training loops](https://docs.kidger.site/equinox/tricks/),
    slightly reducing JAX's overhead when repeatedly passing through the
    jit boundary of the `make_step` function in a training loop.
    It is furthermore compatible with all optimizers from the optax library.
    After each update, any constraints in the model are [applied][klax.apply].

    Args:
        state: [TrainingState][klax.TrainingState].
        batch: Batch of training data.
        static: [TrainingStatic][klax.TrainingStatic].

    Returns:
        Updated training state.

    """
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

    return TrainingState(
        model_leaves=static.disassemble_model(model),
        opt_state_leaves=static.disassemble_opt_state(opt_state),
    )


def run_training_loop(
    view: TrainingView,
    callbacks: Sequence[Callback],
) -> TrainingView:
    """Iterate [`make_step`][klax.make_step] in pure python with callback integration.

    Args:
        view: [TrainingView][klax.TrainingView]
        callbacks: Sequence of [Callback][klax.Callback] instances.

    Returns:
        Final [TrainingView][klax.TrainingView].

    """
    step = 0
    for callback in callbacks:
        callback.on_training_start(view, step)

    state = view._state
    static = view._static
    for step in range(1, static.steps + 1):
        state = make_step(state, next(static.batch), static)

        view = TrainingView(state, static)
        stop = False
        for callback in callbacks:
            stop |= bool(callback.on_training_step(view, step))
        if stop:
            break

    for callback in callbacks:
        callback.on_training_end(view, step)

    view = TrainingView(state, static)
    return view


def fit[T: eqx.Module](
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
    metrics: Sequence[Metric] | None = None,
    log_every: int = 100,
    verbose: Literal[0, 1, 2] = 2,
    callbacks: Sequence[Callback] | None = None,
    key: PRNGKeyArray,
) -> tuple[T, History]:
    """Train a model using an optimizer from optax.

    This is a convenient wrapper around [`firun_training_loopt`][klax.run_training_loop]
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
        validation_data: Arbitrary `PyTree` used for validation during
            training. Must have the same tree structure as `data`. (Defaults
            to None.)
            Internally, the validation data is used to create a [BatchMetric][klax.BatchMetric]
            for logging. Each time the metric is evaluated, the loss is computed
            on a batch from the validation dataset with batch size `4*batch_size`.
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
        metrics: Sequence of [metrics][klax.Metric ] to be evaluated at regular
            intervals during the training. You can overwrite the default "loss"
            and "validation_loss" metrics, by adding custom metrics with the same
            name.
            (Defaults to `None`.)
        log_every: Interval for both metric evaluation and progress logging.
            A value `log_every=n` means that every `n` steps during the training
            the metrics are evaluated and (if `verbose>0`) the training progress
            and selected metric values are printed.
        verbose: Integer controlling the verbosity during training.
            - 0: Nothing is printed.
            - 1: A message is printed every `log_every` steps.
            - 2: A progressbar is used and updated every `log_every` steps.
        callbacks: List of [Callbacks][klax.Callback]. They can be used to
            implement early stopping, custom logging and more. The argument
            to the callback function is aCallbackArgs object.
            (Defaults to `None`.)
        key: A `jax.random.PRNGKey` used to provide randomness for batch
            generation.

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
    view = make_view(
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

    view = run_training_loop(view, callbacks)

    model = view._static.assemble_model(view._state.model_leaves)

    history = logger.history if logger is not None else History()

    return model, history
