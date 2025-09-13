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
import jax.random as jr
import numpy as np
import optax
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from ._datahandler import broadcast_and_get_size
from ._wrappers import apply


@dataclass
class TrainingState:
    flat_model: PyTree
    flat_opt_state: PyTree
    treedef_model: PyTree
    treedef_opt_state: PyTree
    step: int = 0

    def __init__(self, model: PyTree, opt_state: PyTree = None, step: int = 0):
        # Apply the Constraint in the model to ensure apply-constrains are met
        # initially
        model = apply(model)

        # Use the unflatten trick to speed up training,
        # see https://docs.kidger.site/equinox/tricks/
        flat_model, treedef_model = jax.tree.flatten(model)
        flat_opt_state, treedef_opt_state = jax.tree.flatten(opt_state)

        self.flat_model = flat_model
        self.flat_opt_state = flat_opt_state
        self.treedef_model = treedef_model
        self.treedef_opt_state = treedef_opt_state
        self.step = step

    @property
    def model(self) -> PyTree:
        return jax.tree_util.tree_unflatten(
            self.treedef_model, self.flat_model
        )

    @property
    def opt_state(self) -> PyTree:
        return jax.tree_util.tree_unflatten(
            self.treedef_opt_state, self.flat_opt_state
        )

    def update(
        self, flat_model: PyTree, flat_opt_state: PyTree, step: int
    ) -> None:
        self.flat_model = flat_model
        self.flat_opt_state = flat_opt_state
        self.step = step


class Callback(ABC):
    """An abstract callback.

    Inherit from this class to create a custom callback.
    """

    def __call__(self, state: TrainingState) -> bool | None:
        """Call after each step during training."""
        pass

    def on_training_end(self, state: TrainingState) -> None:
        """Call when training ends."""
        pass

    def on_training_start(self, state: TrainingState) -> None:
        """Call when training starts."""
        pass


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
        _, grad = value_and_grad_fn(model, batch)
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


@typing.runtime_checkable
class Batcher(Protocol):
    @abstractmethod
    def __call__(
        self,
        data: PyTree[Any],
        batch_size: int,
        batch_axis: int,
        *,
        key: PRNGKeyArray,
    ) -> Generator[PyTree[Any], TrainingState, None]:
        pass


def stateful_batch_data(
    data: PyTree[Any],
    batch_size: int,
    batch_axis: int,
    convert_to_numpy: bool = True,
    *,
    key: PRNGKeyArray,  # Only cor compliance with the `Batcher` protocol
) -> Generator[PyTree[Any], TrainingState, None]:
    """Create a stateful batch generator that uses the step as seed."""
    batch_axis, dataset_size = broadcast_and_get_size(data, batch_axis)

    # Convert to Numpy arrays. Numpy's slicing is much faster than JAX's, so
    # for fast model training steps this actually makes a huge difference!
    # However, be aware that this is likely only true if JAX runs on CPU.
    if convert_to_numpy:
        data = jax.tree.map(
            lambda x, a: x if a is None else np.array(x),
            data,
            batch_axis,
            is_leaf=lambda x: x is None,
        )

    # Reduce batch size if the dataset has less examples than batch size
    batch_size = min(batch_size, dataset_size)

    indices = jnp.arange(dataset_size)
    while True:
        # Store the training state as received by the `.send(state)` within
        # the training loop.
        state: TrainingState = yield
        key = jax.random.PRNGKey(state.step)  # Create key from step
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)  # Update key
        start, end = 0, batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield jax.tree.map(
                lambda a, x: x if a is None else x[batch_perm],
                batch_axis,
                data,
                is_leaf=lambda x: x is None,
            )
            start = end
            end = start + batch_size


def fit_core[T: eqx.Module](
    updater: Updater,
    batcher: Generator[PyTree[Any], TrainingState, None],
    state: TrainingState,
    steps: int,
):
    @eqx.filter_jit
    def make_step(batch, flat_model, flat_opt_state):
        # Use the unflatten trick to speed up training,
        # see https://docs.kidger.site/equinox/tricks/
        model = jax.tree_util.tree_unflatten(state.treedef_model, flat_model)
        opt_state = jax.tree_util.tree_unflatten(
            state.treedef_opt_state, flat_opt_state
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

    for state.step in range(state.step, state.step + steps + 1):
        next(batcher)
        batch = batcher.send(state)  # Send the state back to the batcher
        state.flat_model, state.flat_opt_state = make_step(
            batch, state.flat_model, state.flat_opt_state
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
    batcher: Batcher = stateful_batch_data,
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
    batcher = stateful_batch_data(
        (x, y),
        batch_size=32,
        batch_axis=batch_axis,
        key=eqx.internal.GetKey()(),  # Unused
    )
    loss = mse(batch_axis)
    updater = optax_transform_update_fn_updater(optimizer.update, *loss)
    state = fit_core(updater, batcher, state, steps=1000)
    y_pred = jax.vmap(state.model)(x)
    assert jnp.allclose(y_pred, y)

    import pprint

    pprint.pp(state)
