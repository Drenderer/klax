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

import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from ._datahandler import BatchGenerator, batch_data
from ._wrappers import apply


@dataclass
class TrainingState:
    model: PyTree
    opt_state: PyTree
    step: int = 0


@typing.runtime_checkable
class ValueFn(Protocol):
    @abstractmethod
    def __call__(self, model: PyTree, data: PyTree) -> Scalar:
        pass


@typing.runtime_checkable
class ValueAndGradFn(Protocol):
    @abstractmethod
    def __call__(self, model: PyTree, data: PyTree) -> tuple[Scalar, PyTree]:
        pass


class LossFactory(Protocol):
    @abstractmethod
    def __call__(self, batch_axis) -> tuple[ValueFn, ValueAndGradFn]:
        pass


def mse(batch_axis) -> tuple[ValueFn, ValueAndGradFn]:
    def value_fn(model: PyTree, data: PyTree) -> Scalar:
        x, y = data
        if isinstance(batch_axis, tuple):
            in_axes = batch_axis[0]
        else:
            in_axes = batch_axis
        y_pred = eqx.filter_vmap(model, in_axes=(in_axes,))(x)
        return jnp.mean(jnp.square(y_pred - y))

    def value_and_grad_fn(
        model: PyTree, data: PyTree
    ) -> tuple[Scalar, PyTree]:
        return eqx.filter_value_and_grad(value_fn)(model, data)

    return value_fn, value_and_grad_fn


class Updater(Protocol):
    @abstractmethod
    def __call__(
        self, model: PyTree, batch: PyTree, opt_state: PyTree
    ) -> tuple[PyTree, PyTree]:
        pass


class UpdaterFactory(Protocol):
    @abstractmethod
    def __call__(
        self,
        opt_update: optax.TransformUpdateFn | optax.TransformUpdateExtraArgsFn,
        value_fn: ValueFn,
        value_and_grad_fn: ValueAndGradFn,
    ) -> Updater:
        pass


def optax_transform_update_fn_updater(
    opt_update: optax.TransformUpdateFn,
    value_fn: ValueFn,
    value_and_grad_fn: ValueAndGradFn,
) -> Updater:
    def wrapper(model, batch, opt_state):
        value, grad = value_and_grad_fn(model, batch)
        updates, opt_state = opt_update(
            grad,
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    return wrapper


def optax_transform_update_fn_extra_args_updater(
    opt_update: optax.TransformUpdateExtraArgsFn,
    value_fn: ValueFn,
    value_and_grad_fn: ValueAndGradFn,
) -> Updater:
    def wrapper(model, batch, opt_state):
        value, grad = value_and_grad_fn(model, batch)
        updates, opt_state = opt_update(
            grad,
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
            value=value,
            grad=grad,
            value_fn=jax.tree_util.Partial(value_fn, model=model, batch=batch),
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    return wrapper


def fit_core[T: eqx.Module](
    updater: Updater,
    batcher: Generator[PyTree[Any], None, None],
    state: TrainingState,
    steps: int,
):
    @eqx.filter_jit
    def make_step(batch, flat_model, flat_opt_state):
        # Use the unflatten trick to speed up training,
        # see https://docs.kidger.site/equinox/tricks/
        model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
        opt_state = jax.tree_util.tree_unflatten(
            treedef_opt_state, flat_opt_state
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

    # Apply the Constraint in the model to ensure apply-constrains are met
    # initially
    state.model = apply(state.model)

    # Use the unflatten trick to speed up training,
    # see https://docs.kidger.site/equinox/tricks/
    flat_model, treedef_model = jax.tree.flatten(state.model)
    flat_opt_state, treedef_opt_state = jax.tree.flatten(state.opt_state)

    for state.step in range(state.step, state.step + steps + 1):
        batch = next(batcher)
        flat_model, flat_opt_state = make_step(
            batch, flat_model, flat_opt_state
        )

    state.model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
    state.opt_state = jax.tree_util.tree_unflatten(
        treedef_opt_state, flat_opt_state
    )

    return state


def fit(
    model,
    data,
    *,
    batch_size: int = 32,
    batch_axis: PyTree[int | None] = 0,
    steps: int = 1_000,
    loss: LossFactory = mse,
    optimizer: optax.GradientTransformation,
    init_opt_state: PyTree[Any] = None,
    batcher: BatchGenerator = batch_data,
    updater: UpdaterFactory = optax_transform_update_fn_updater,
    key: PRNGKeyArray,
):
    state = TrainingState(
        model=model,
        opt_state=optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        if init_opt_state is None
        else init_opt_state
        if init_opt_state is None
        else init_opt_state,
    )

    state = fit_core(
        updater(optimizer.update, *loss(batch_axis)),
        batcher(
            data=data, batch_axis=batch_axis, batch_size=batch_size, key=key
        ),
        state,
        steps,
    )
    return state.model


if __name__ == "__main__":
    # Test fit
    x = jnp.linspace(0.0, 1.0, 2).reshape(-1, 1)
    y = 2.0 * x + 1.0
    model = eqx.nn.Linear(1, 1, key=eqx.internal.GetKey()())
    model = fit(
        model, (x, y), optimizer=optax.adam(1.0), key=eqx.internal.GetKey()()
    )
    y_pred = jax.vmap(model)(x)
    assert jnp.allclose(y_pred, y)

    # Test fit_core
    x = jnp.linspace(0.0, 1.0, 2).reshape(-1, 1)
    y = 2.0 * x + 1.0
    model = eqx.nn.Linear(1, 1, key=eqx.internal.GetKey()())
    batch_axis = 0
    optimizer = optax.adam(1.0)
    state = TrainingState(
        model=model,
        opt_state=optimizer.init(eqx.filter(model, eqx.is_inexact_array)),
    )
    batcher = batch_data(
        (x, y),
        batch_size=32,
        batch_axis=batch_axis,
        key=eqx.internal.GetKey()(),
    )
    loss = mse(batch_axis)
    updater = optax_transform_update_fn_updater(optimizer.update, *loss)
    state = fit_core(updater, batcher, state, steps=1000)
    y_pred = jax.vmap(state.model)(x)
    assert jnp.allclose(y_pred, y)
