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

from collections.abc import Callable, Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import optax
from jaxtyping import PRNGKeyArray, PyTree

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

param_spec = eqx.is_inexact_array


def make_step(
    state_leaves: list[Leaf],
    state_treedef: Any,
    batch: PyTree,
    loss: Loss,
    optimizer: optax.GradientTransformationExtraArgs,
) -> list[Leaf]:
    state = jax.tree.unflatten(state_treedef, state_leaves)

    model_params, model_static = eqx.partition(state.model, param_spec)
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

    state = TrainingState(model, opt_state, state.run_state)
    state_leaves, _ = jax.tree.flatten(state)

    return state_leaves


def run_training_loop(
    context: TrainingContext,
    callbacks: Sequence[Callback],
    step_function: Callable,
) -> TrainingContext:
    state_leaves = context._state_leaves
    state_treedef = context._state_treedef

    for callback in callbacks:
        callback.on_training_start(context)

    for batch in context.batch_generator:
        if context.step >= context.steps:
            break

        state_leaves = step_function(
            state_leaves,
            state_treedef,
            batch,
            context.loss,
            context.optimizer,
        )
        step = context.step + 1
        context.update(state_leaves, step)

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
    batch_axes: PyTree[int | str | None] = 0,
    run_state: PyTree[Any] = None,
    validation_data: PyTree[Any] = None,
    steps: int = 1000,
    loss: Loss,
    optimizer: optax.GradientTransformation
    | optax.GradientTransformationExtraArgs = optax.adam(1e-3),
    init_opt_state: PyTree[Any] = None,
    batcher: Batcher = batch_data,
    make_logger: bool = True,
    metrics: Sequence[Metric] | None = None,
    log_every: int = 100,
    verbose: Literal[0, 1, 2] = 2,
    callbacks: Sequence[Callback] | None = None,
    jit_compile: bool = True,
    vmap_ensemble: bool = False,
    key: PRNGKeyArray,
) -> tuple[T, History | None]:
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
            arrays in `data`. Each leaf is one of `int` (positional axis on
            an array leaf), `str` (dim name on an `xarray` leaf), or `None`
            (the corresponding subtree is not batched). `batch_axes` must
            be a prefix of `data`. By specifying `batch_axes` as a `PyTree`
            it is possible to specify different batch axes for different
            leaves of `data`. (Defaults to `0`, meaning the first axes of
            arrays in `data` are batch dimensions. xarray leaves require
            an explicit `str` dim name.)

            !!!Example
                For a dataset of 100 examples `data = (x, (y1, y2), "some_string")`
                where `x` has  shape `(32, 100)`, `y1` has shape `(100,)`
                and `y2` has shape `(100, 10)`, an appropriate `batch_axes`
                would be `batch_axes = (1, 0, None)` indicating that
                the batch axis for `x` is the second axis, for `y1` and `y2`
                the first axis and that there is no batch axis for the string.
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
        make_logger: Wether to create a [`MetricLogger`][klax.MetricLogger].
            If `False` the arguments `metrics`, `log_every` and `verbose`
            don't have any effect and `fit` will return `None` instead of a
            `History`. This is useful for implementing custom logging.
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
        jit_compile: Wether to compile the function that computes the gradient
            update step (this includes the loss function). You generally want
            this to be `True`. However, in cases where the loss function itself
            is not jit-able (e.g., when computing different parts of the
            loss on different hardware, such as GPU and CPU) it might be
            advantageous to have more fine grained control over the compilation.
        vmap_ensemble: If true, the step function and optimizer state
            initialization are vmapped across the leading axis of the arrays in
            the [`TrainingState`][klax.TrainingState], i.e., model, optimizer
            state and run state.
            This is useful to train multiple instances of the same model
            (ensemble) in a single call to `fit`.

            !!!Note
                The batcher is **not** vmapped, meaning that all models in the
                ensemble will receive identical batches of data.

            !!!Example
                ```python
                    @eqx.filter_vmap
                    def make_mlp_ensemble(key):
                        return klax.nn.MLP("scalar", "scalar", [16, 16], key=key)

                    mlp_ensemble = make_ensemble(jr.split(key, 10))

                    mlp_ensemble, history = klax.fit(mlp_ensemble, ..., vmap_ensemble=True, ...)

                    @eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
                    def evaluate_ensemble(ensemble, x):
                        return ensemble(x)

                    evaluate_ensemble(mlp_ensemble, jax.random.normal(key, (2,)))
                ```
        key: A `jax.random.PRNGKey` used to provide randomness for batch
            generation.

    Returns:
        A tuple of the trained model and the training history.

    Note:
        The returned history will be `None` if `make_logger=False`.

    """
    # Transform the step function
    step_function = make_step
    if vmap_ensemble:
        step_function = eqx.filter_vmap(
            step_function, in_axes=(eqx.if_array(0), None, None, None, None)
        )
    if jit_compile:
        step_function = eqx.filter_jit(step_function)

    if init_opt_state is None:
        # Initialize the optimizer and 'tell it' to optimize with respect to
        # all inexact arrays in the model. This is done by passing the model to
        # the optimizer.
        if vmap_ensemble:
            opt_state = jax.vmap(optimizer.init)(eqx.filter(model, param_spec))
        else:
            opt_state = optimizer.init(eqx.filter(model, param_spec))
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

    if make_logger:
        # Initialize logging and default metrics
        metric_func = (
            eqx.filter_vmap(loss, in_axes=(eqx.if_array(0), None, None))
            if vmap_ensemble
            else loss
        )
        _metrics = []
        _metrics.append(
            BatchMetric(
                "loss",
                metric_func,
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
                    metric_func,
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

    context = run_training_loop(
        context, callbacks, step_function=step_function
    )

    model = context.state.model

    if make_logger:
        return model, logger.history

    return model, None
