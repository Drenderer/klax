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

"""Implements a training loop."""

from collections.abc import Callable, Generator, Iterable
from typing import Any, Protocol, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import PRNGKeyArray, PyTree, Scalar
from optax._src.utils import Sequence

from ._callbacks import Callback, HistoryCallback
from ._context import (
    EvaluationContext,
    TimingInfo,
    TrainingContext,
    TrainingState,
)
from ._datahandler import Batcher, BatchGenerator, batch_data
from ._losses import LossFactory, mse
from ._updaters import (
    Updater,
    UpdaterFactory,
    optax_transform_update_fn_updater,
)
from ._wrappers import apply


def fit_core(
    updater: Updater,
    batcher: BatchGenerator,
    ctx: TrainingContext,
    steps: int,
    callbacks: Iterable[Callback] | Callback | None = None,
) -> tuple[TrainingContext, list[Callback]]:
    @eqx.filter_jit
    def make_step(batch, flat_model, flat_opt_state):
        # Use the unflatten trick to speed up training,
        # see https://docs.kidger.site/equinox/tricks/
        model = jax.tree_util.tree_unflatten(ctx.treedef_model, flat_model)
        opt_state = jax.tree_util.tree_unflatten(
            ctx.treedef_opt_state, flat_opt_state
        )

        # Compute and apply the parameter updates
        # params, static = eqx.partition(model, eqx.is_inexact_array)
        # params = updater(params, static, batch, opt_state)
        model, opt_state = updater(model, batch, opt_state)

        # Apply the Constraint in the model to ensure apply-constrains are met
        # after the update.
        model = apply(model)

        flat_model = jax.tree_util.tree_leaves(model)
        flat_opt_state = jax.tree_util.tree_leaves(opt_state)

        return flat_model, flat_opt_state

    # Make callbacks iterable
    if callbacks is None:
        callbacks = []
    elif isinstance(callbacks, Callback):
        callbacks = [callbacks]
    else:
        callback = list(callbacks)

    # Initialize context and callbacks
    ctx.update(ctx.flat_model, ctx.flat_opt_state)
    for callback in callbacks:
        callback.on_training_start(ctx)

    for _ in range(steps):
        next(batcher)  # Prime the batcher
        batch = batcher.send(ctx)  # Send the context

        flat_model, flat_opt_state = make_step(
            batch, ctx.flat_model, ctx.flat_opt_state
        )

        ctx.update(flat_model, flat_opt_state)

        # Run all callbacks and break if any of them request termination of
        # the training loop.
        # Note! The square brackets are important. Otherwise the loop is
        # terminated with the first callback that returns true. But we want
        # to run all callbacks first and then decide, whether to terminate.
        if any([callback(ctx) for callback in callbacks]):
            break

    # Call callbacks after training
    ctx.update(ctx.flat_model, ctx.flat_opt_state)
    for callback in callbacks:
        callback.on_training_end(ctx)

    return ctx, list(callbacks)


def fit[T: eqx.Module](
    model: T,
    data,
    *,
    batch_size: int = 32,
    batch_axis: PyTree[int | None] = 0,
    validation_data: PyTree[Any] = None,
    steps: int = 1_000,
    loss: LossFactory = mse,
    optimizer: optax.GradientTransformation,
    init_opt_state: PyTree[Any] = None,
    batcher: Batcher = batch_data,
    updater: UpdaterFactory = optax_transform_update_fn_updater,
    callbacks: Iterable[Callback] | Callback | None = HistoryCallback(),
    key: PRNGKeyArray,
) -> tuple[T, list[Callback]]:
    value_fn, value_and_grad_fn = loss(batch_axis)
    evaluator = EvaluationContext(value_fn, data, val_data=validation_data)
    state = TrainingState(
        model=model,
        opt_state=optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        if init_opt_state is None
        else init_opt_state
        if init_opt_state is None
        else init_opt_state,
    )
    ctx = TrainingContext(
        state=state,
        evaluator=evaluator,
        timing=TimingInfo(),
    )

    ctx, callbacks = fit_core(
        updater(optimizer.update, value_fn, value_and_grad_fn),
        batcher(
            data=data, batch_axis=batch_axis, batch_size=batch_size, key=key
        ),
        ctx,
        steps,
        callbacks=callbacks,
    )
    return ctx.model, callbacks


if __name__ == "__main__":
    # Test fit
    x = jnp.linspace(0.0, 1.0, 2).reshape(-1, 1)
    y = 2.0 * x + 1.0
    model = eqx.nn.Linear(1, 1, key=eqx.internal.GetKey()())
    model, _ = fit(
        model, (x, y), optimizer=optax.adam(1.0), key=eqx.internal.GetKey()()
    )
    y_pred = jax.vmap(model)(x)
    assert jnp.allclose(y_pred, y)

    # Test fit_core
    x = jnp.linspace(0.0, 1.0, 2).reshape(-1, 1)
    y = 2.0 * x + 1.0
    data = (x, y)
    model = eqx.nn.Linear(1, 1, key=eqx.internal.GetKey()())
    batch_axis = 0
    optimizer = optax.adam(1.0)
    value_fn, value_and_grad_fn = mse(batch_axis)
    evaluator = EvaluationContext(value_fn, (x, y))
    state = TrainingState(
        model=model,
        opt_state=optimizer.init(eqx.filter(model, eqx.is_inexact_array)),
    )
    ctx = TrainingContext(
        state=state, evaluator=evaluator, timing=TimingInfo()
    )
    batcher = batch_data(
        (x, y),
        batch_size=32,
        batch_axis=batch_axis,
        key=eqx.internal.GetKey()(),  # Unused
    )
    updater = optax_transform_update_fn_updater(
        optimizer.update, value_fn, value_and_grad_fn
    )
    ctx, _ = fit_core(updater, batcher, ctx, steps=1000)
    y_pred = jax.vmap(ctx.model)(x)
    assert jnp.allclose(y_pred, y)

    import pprint

    pprint.pp(state)
