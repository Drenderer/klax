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

"""
This module implements a basic training loop.
"""

from __future__ import annotations
from typing import Any, Iterable, Optional

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, PyTree
import optax

from ._callbacks import (
    Callback,
    CallbackArgs,
    HistoryCallback,
)
from ._datahandler import batch_data, BatchGenerator, broadcast_and_get_batch_size
from ._losses import Loss, mse
from ._wrappers import unwrap


def fit[T: eqx.Module, H: Callback](
    model: T,
    data: PyTree[Any],
    *,
    batch_size: int = 32,
    batch_axis: PyTree[int | None] = 0,
    validation_data: Optional[PyTree[Any]] = None,
    steps: int = 1000,
    loss_fn: Loss = mse,
    optimizer: optax.GradientTransformation = optax.adam(1e-3),
    init_opt_state: PyTree[Any] = None,
    batcher: BatchGenerator = batch_data,
    history: Optional[H] = None,
    callbacks: Optional[Iterable[Callback]] = None,
    key: PRNGKeyArray,
) -> tuple[T, HistoryCallback | H]:
    """Trains a model using an optimizer from optax.

    Args:
        model: The model instance, which should be trained. It must be a subclass of
            `eqx.Module`. The model may contain :class:`klax.AbstractUnwrappable` wrappers.
        data: The training data can be any `PyTree` with `ArrayLike` leaves.
            Most likely you'll want `data` to be a tuple `(x, y)` with model inputs
            `x` and model outputs `y`.
        batch_size: The number of examples in a batch.
        batch_axis: A `PyTree` denoting, which axis is the batch axis for arrays in `data`.
            `batch_axis` must be a prefix of `data`. By specifying `batch_axis` as a `PyTree`
            it is possible to specify different batch axes for different leaves of `data`.
            (Defaults to `0`, meaning the first axis of arrays in `data` are batch dimensions.)
        validation_data: Arbitrary `PyTree` used for validation during training.
            Must have the same tree structure as `data`.
            (Defaults to None. Keyword only argument)
        steps: Number of gradient updates to apply. (Defaults to 1000. Keyword only argument)
        loss_fn: The loss function with call signature
            `(model: PyTree, data: PyTree, batch_axis: int | None | Sequence[Any]) -> float`.
            (Defaults to `mse`.)
        optimizer: The optimizer. Any optax gradient transform to calculate the updates for
            the model. (Defaults to optax.adam(1e-3).)
        init_opt_state: The initial state of the optimizer. If `None`, the optimizer is initialized
            from scratch. By providing a value for `init_opt_state`, the user can resume training from a
            previous state (e.g., obtained from the `HistoryCallback.last_opt_state`).
            (Defaults to `None`.)
        batcher: The data loader that splits inputs and targets into batches.
            (Defaults to `batch_data`.)
        History: A callback intended for tracking the training process.
            If no custom callback is passed the :obj:`klax.HistoryCallback` with a logging interval of
            100 steps is used. To change the logging increment or verbosity of this default callback,
            pass a `HistoryCallback` object to this argument, e.g.,
            `history=HistoryCallback(log_every=10, verbose=False)` for logging on every 10-th step
            without printing the loss.
        callbacks: Callback functions that are evaluated after every training step. They can
            be used to implement early stopping, custom history logging and more. The argument to the
            callback function is a CallbackArgs object. (Defaults to `None`. Keyword only Argument)
        key: A `jax.random.PRNGKey` used to provide randomness for batch generation.
            (Keyword only argument.)

    Note:
        This function assumes that the batch dimension is always oriented along
        the first axes of any `jax.Array`

    Returns:
        A tuple of the trained model and a history dictionary containing the loss history.
    """

    # Braodcast the batch_axis to the data. While this happens again in the batch_data,
    # doing it here allows the use of the broadcasted batch_axis in the loss function.
    # If `batch_axis` is a prefix of `data`, this ensures that only leafs of
    # type ArrayLike are vmapped. Thus it is possible to have data like `(str, array)`
    # ans still use `batch_axis=0` instead of `batch_axis=(None, 0)`.
    batch_axis, dataset_size = broadcast_and_get_batch_size(data, batch_axis)

    # Define a function to calculate the loss. This is jit compiled to speed up
    # the loss evaluation for the loss history.
    @eqx.filter_jit
    def combined_loss(model, batch):
        model = unwrap(model)
        return loss_fn(model, batch, batch_axis=batch_axis)

    # This partitioned loss function is required within the make_step function, because
    # the optax.lbgfs GradientTransformation required the loss function to be diretly
    # dependent on the parameters.
    def partitioned_loss(params, static, batch):
        model = eqx.combine(params, static)
        return combined_loss(model, batch)

    @eqx.filter_jit
    def make_step(batch, flat_model, optimizer, flat_opt_state):
        # Use the unflatten trick to speed up training,
        # see https://docs.kidger.site/equinox/tricks/
        model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
        opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, flat_opt_state)

        # Compute and apply the parameter updates
        params, static = eqx.partition(model, eqx.is_inexact_array)
        value, grad = jax.value_and_grad(partitioned_loss)(params, static, batch)
        updates, opt_state = optimizer.update(
            grad,
            opt_state,
            params,
            value=value,
            grad=grad,
            value_fn=jax.tree_util.Partial(partitioned_loss, static=static, batch=batch)
        )
        params = optax.apply_updates(params, updates)
        model = eqx.combine(params, static)

        flat_model = jax.tree_util.tree_leaves(model)
        flat_opt_state = jax.tree_util.tree_leaves(opt_state)

        return flat_model, flat_opt_state

    if init_opt_state is None:
        # Initialize the optimizer and 'tell it' to optimize with respect to all
        # inexact arrays in the model. This is done by passing the model to the optimizer.
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    else:
        opt_state = init_opt_state

    # Use the unflatten trick to speed up training,
    # see https://docs.kidger.site/equinox/tricks/
    flat_model, treedef_model = jax.tree.flatten(model)
    flat_opt_state, treedef_opt_state = jax.tree.flatten(opt_state)

    # Make callbacks iterable
    callbacks = [] if callbacks is None else list(callbacks)

    # Initialize callback arguments and history
    if history is None:
        history = HistoryCallback(log_every=100)
    callbacks.append(history)

    cbargs = CallbackArgs(
        combined_loss, treedef_model, treedef_opt_state, data, validation_data
    )

    # Call callbacks after training
    cbargs.update(flat_model, flat_opt_state, 0)
    for callback in callbacks:
        callback.on_training_start(cbargs)

    # Loop over all training steps
    for step, batch in zip(
        range(1, steps + 1),
        batcher(data, batch_size, batch_axis, key=key),
    ):
        flat_model, flat_opt_state = make_step(
            batch, flat_model, optimizer, flat_opt_state
        )

        # Update callbacks arguments with the current state of the model
        cbargs.update(flat_model, flat_opt_state, step)

        # Run all callbacks and break if any of them request termination of
        # the training loop.
        # Note! The square brackets are important. Otherwise the loop is
        # terminated with the first callback that returns true. But we want
        # to run all callbacks first and then decide, whether to terminate.
        if any([callback(cbargs) for callback in callbacks]):
            break

    model = jax.tree_util.tree_unflatten(treedef_model, flat_model)

    # Call callbacks after training
    cbargs.update(flat_model, flat_opt_state, -1)
    for callback in callbacks:
        callback.on_training_end(cbargs)

    return model, history
