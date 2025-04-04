"""
This module implements a basic training loop.
"""

from __future__ import annotations
import datetime
import time
from typing import Any, Iterable, Optional

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, PyTree
import numpy as np
import optax
import paramax as px

from .callbacks import Callback, CallbackArgs
from .datahandler import dataloader, Dataloader, broadcast_and_get_batch_size
from .losses import Loss, mse



def fit[T: eqx.Module](
    model: T,
    data: PyTree[Any],
    *,
    batch_size: int = 32,
    batch_axis: PyTree[int | None] = 0,
    validation_data: Optional[PyTree[Any]] = None,
    steps: int = 1000,
    log_every: int = 100,
    loss_fn: Loss = mse,
    optimizer: optax.GradientTransformation = optax.adam(1e-3),
    dataloader: Dataloader = dataloader,
    callbacks: Optional[Iterable[Callback]] = None,
    key: PRNGKeyArray,
) -> tuple[T, dict]:
    """Trains a model using an optimizer from optax.

    Args:
        model: The model instance, which should be trained. It must be a subclass of
            `eqx.Module`. The model may contain `paramax.AbstractUnwrappable` wrappers.
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
        log_every: The number of steps between updates of the loss history. A history update
            consists of calculating the training and validation losses *over the entire datasets*
            and storing them in the history dictionary. (Defaults to 10. Keyword only Argument)
        loss_fn: The loss function with call signature `(model, prediction, target, in_axes) -> float`.
            (Defaults to `mse`.)
        optimizer: The optimizer. Any optax gradient transform to calculate the updates for
            the model. (Defaults to optax.adam(1e-3).)
        dataloader: The data loader that splits inputs and targets into batches.
            (Defaults to `dataloader`)
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

    # Braodcast the batch_axis to the data. While this happens again in the dataloader,
    # doing it here allows the use of the broadcasted batch_axis in the loss function.
    # If `batch_axis` is a prefix of `data`, this ensures that only leafs of 
    # type ArrayLike are vmapped. Thus it is possible to have data like `(str, array)`
    # ans still use `batch_axis=0` instead of `batch_axis=(None, 0)`. 
    batch_axis, dataset_size = broadcast_and_get_batch_size(data, batch_axis)

    # Define a function to calculate the loss. This is jit compiled to speed up
    # the loss evaluation for the loss history.
    @eqx.filter_jit
    def get_loss(model, batch):
        model = px.unwrap(model)
        return loss_fn(model, batch, batch_axis=batch_axis)

    # Get the gradient function
    grad_loss = eqx.filter_grad(get_loss)

    @eqx.filter_jit
    def make_step(batch, flat_model, optimizer, flat_opt_state):
        # Use the unflatten trick to speed up training,
        # see https://docs.kidger.site/equinox/tricks/
        model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
        opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, flat_opt_state)

        # Compute and apply the parameter updates
        grads = grad_loss(model, batch)
        updates, opt_state = optimizer.update(
            grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)

        flat_model = jax.tree_util.tree_leaves(model)
        flat_opt_state = jax.tree_util.tree_leaves(opt_state)

        return flat_model, flat_opt_state

    # Initialize the history dict
    history = {
        "steps": [],
        "loss": [],
        "training_time": 0.0
    }
    if validation_data is not None:
        history["val_loss"] = []

    val_loss = None

    # Initialize the optimizer and 'tell it' to optimize with respect to all
    # inexact arrays in the model
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Use the unflatten trick to speed up training,
    # see https://docs.kidger.site/equinox/tricks/
    flat_model, treedef_model = jax.tree_util.tree_flatten(model)
    flat_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state)

    cbargs = CallbackArgs(get_loss, data, validation_data, treedef_model)

    # Loop over all training steps
    start_time = time.time()
    for step, batch in zip(
        range(1, steps + 1),
        dataloader(
            data,
            batch_size,
            batch_axis,
            key=key,
        ),
    ):
        flat_model, flat_opt_state = make_step(
            batch, flat_model, optimizer, flat_opt_state
        )

        # Update callbacks arguments with the current state of the model
        cbargs.update(flat_model, step)

        # Log every log_every steps and the last step
        if (step % log_every) == 0 or step == steps:
            loss = cbargs.loss
            history["steps"].append(cbargs.step)
            history["loss"].append(loss)
            message = f"Step: {step}, Loss: {loss:.3e}"

            if validation_data is not None:
                val_loss = cbargs.val_loss
                history["val_loss"].append(val_loss)
                message += f", Validation loss: {val_loss:.3e}"

            print(message)

        if callbacks is not None:
            # Run all callbacks and break if any of them request termination of
            # the training loop.
            # Note! The square brackets are important. Otherwise the loop is
            # terminated with the first callback that returns true. But we want
            # to run all callbacks first and then decide, whether to terminate.
            if any([callback(cbargs) for callback in callbacks]):
                break

    model = jax.tree_util.tree_unflatten(treedef_model, flat_model)

    training_time = time.time() - start_time
    print(f"Training took: {datetime.timedelta(seconds=training_time)}")
    history["training_time"] = training_time

    # Convert history to numpy arrays
    history = {k: np.array(v) for k, v in history.items()}

    return model, history
