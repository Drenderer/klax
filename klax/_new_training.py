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
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from ._datahandler import broadcast_and_get_size
from ._wrappers import apply


class TrainingState:
    _flat_model: PyTree
    _flat_opt_state: PyTree
    _treedef_model: PyTree
    _treedef_opt_state: PyTree
    _step: int
    _cache: dict[str, Any] = {}

    def __init__(
        self, model: PyTree, opt_state: PyTree = None, initial_step: int = 0
    ):
        # Apply the Constraint in the model to ensure apply-constrains are met
        # initially
        model = apply(model)

        # Use the unflatten trick to speed up training,
        # see https://docs.kidger.site/equinox/tricks/
        flat_model, treedef_model = jax.tree.flatten(model)
        flat_opt_state, treedef_opt_state = jax.tree.flatten(opt_state)

        self._flat_model = flat_model
        self._flat_opt_state = flat_opt_state
        self._treedef_model = treedef_model
        self._treedef_opt_state = treedef_opt_state
        self._step = initial_step

    @staticmethod
    def _lazy_chached_property(fun: Callable) -> property:
        """Turn a public method into a lazily evaluated property.

        The return value of ``fun`` is stored in the ``_cache`` dictionary of
        the current object using the function name as key. If the name is
        already in ``_cache`` then the cached value is simply returned,
        without evaluating ``fun``.

        Args:
            fun: Method to wrap.

        Returns:
            Wrapped method as a property.

        """
        attr_name = fun.__name__

        def wrapper(self: Self):
            if attr_name not in self._cache:
                self._cache.setdefault(attr_name, fun(self))
            return self._cache.get(attr_name)

        wrapper.__doc__ = fun.__doc__

        return property(wrapper)

    @property
    def flat_model(self) -> PyTree:
        return self._flat_model

    @property
    def flat_opt_state(self) -> PyTree:
        return self._flat_opt_state

    @property
    def step(self) -> int:
        return self._step

    @_lazy_chached_property
    def model(self) -> PyTree:
        return jax.tree_util.tree_unflatten(
            self._treedef_model, self._flat_model
        )

    @_lazy_chached_property
    def opt_state(self) -> PyTree:
        return jax.tree_util.tree_unflatten(
            self._treedef_opt_state, self._flat_opt_state
        )

    @property
    def treedef_model(self) -> PyTree:
        return self._treedef_model

    @property
    def treedef_opt_state(self) -> PyTree:
        return self._treedef_opt_state

    def update(self, flat_model: PyTree, flat_opt_state: PyTree):
        self._flat_model = flat_model
        self._flat_opt_state = flat_opt_state
        self._step += self._step

        # Clear cache
        self._cache.clear()


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


@dataclass
class EvaluationContext:
    value_fn: ValueFn
    data: PyTree[Any]
    val_data: PyTree[Any] | None = None
    _cached_step: int | None = None
    _cache: dict[str, Any] = field(default_factory=dict)

    def _ensure_step(self, state: TrainingState):
        if self._cached_step != state.step:
            self._cache.clear()
            self._cached_step = state.step

    @eqx.filter_jit
    def _loss_impl(self, model: PyTree, batch: PyTree[Any]):
        return self.value_fn(model, batch)

    def loss(self, state: TrainingState) -> Scalar:
        self._ensure_step(state)
        if "loss" not in self._cache:
            self._cache["loss"] = self._loss_impl(state.model, self.data)
        return self._cache["loss"]

    def val_loss(self, state: TrainingState) -> Scalar | None:
        self._ensure_step(state)
        if self.val_data is None:
            return None

        if "val_loss" not in self._cache:
            self._cache["val_loss"] = self._loss_impl(
                state.model, self.val_data
            )
        return self._cache["val_loss"]


@dataclass
class TrainingContext:
    state: TrainingState
    evaluator: EvaluationContext

    @property
    def model(self) -> PyTree:
        return self.state.model

    @property
    def optimizer_state(self) -> PyTree:
        return self.state.opt_state

    @property
    def step(self) -> int:
        return self.state.step

    @property
    def loss(self) -> Scalar:
        return self.evaluator.loss(self.state)

    @property
    def val_loss(self) -> Scalar | None:
        return self.evaluator.val_loss(self.state)


class Callback(ABC):
    """An abstract callback.

    Inherit from this class to create a custom callback.
    """

    def __call__(self, ctx: TrainingContext) -> bool | None:
        """Call after each step during training."""
        pass

    def on_training_end(self, ctx: TrainingContext) -> None:
        """Call when training ends."""
        pass

    def on_training_start(self, ctx: TrainingContext) -> None:
        """Call when training starts."""
        pass


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
) -> Generator[PyTree[Any], TrainingContext, None]:
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
        ctx: TrainingContext = yield
        key = jax.random.PRNGKey(ctx.state.step)  # Create key from step
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
    batcher: Generator[PyTree[Any], TrainingContext, None],
    ctx: TrainingContext,
    steps: int,
    callbacks: Iterable[Callback] | None = None,
):
    @eqx.filter_jit
    def make_step(batch, flat_model, flat_opt_state):
        # Use the unflatten trick to speed up training,
        # see https://docs.kidger.site/equinox/tricks/
        model = jax.tree_util.tree_unflatten(
            ctx.state.treedef_model, flat_model
        )
        opt_state = jax.tree_util.tree_unflatten(
            ctx.state.treedef_opt_state, flat_opt_state
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
    callbacks = [] if callbacks is None else list(callbacks)

    for callback in callbacks:
        callback.on_training_start(ctx)

    for _ in range(steps):
        next(batcher)
        batch = batcher.send(ctx)  # Send the context
        flat_model, flat_opt_state = make_step(
            batch, ctx.state.flat_model, ctx.state.flat_opt_state
        )

        ctx.state.update(flat_model, flat_opt_state)

        # Run all callbacks and break if any of them request termination of
        # the training loop.
        # Note! The square brackets are important. Otherwise the loop is
        # terminated with the first callback that returns true. But we want
        # to run all callbacks first and then decide, whether to terminate.
        if any([callback(ctx) for callback in callbacks]):
            break

    # Call callbacks after training
    for callback in callbacks:
        callback.on_training_end(ctx)

    return ctx


def fit(
    model,
    data,
    *,
    batch_size: int = 32,
    batch_axis: PyTree[int | None] = 0,
    validation_data: PyTree[Any] = None,
    steps: int = 1_000,
    loss: LossFactory = mse,
    optimizer: optax.GradientTransformation,
    init_opt_state: PyTree[Any] = None,
    batcher: Batcher = stateful_batch_data,
    updater: UpdaterFactory = optax_transform_update_fn_updater,
    key: PRNGKeyArray,
):
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
    )

    ctx = fit_core(
        updater(optimizer.update, value_fn, value_and_grad_fn),
        batcher(
            data=data, batch_axis=batch_axis, batch_size=batch_size, key=key
        ),
        ctx,
        steps,
    )
    return ctx.state.model


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
        state=state,
        evaluator=evaluator,
    )
    batcher = stateful_batch_data(
        (x, y),
        batch_size=32,
        batch_axis=batch_axis,
        key=eqx.internal.GetKey()(),  # Unused
    )
    updater = optax_transform_update_fn_updater(
        optimizer.update, value_fn, value_and_grad_fn
    )
    state = fit_core(updater, batcher, ctx, steps=1000)
    y_pred = jax.vmap(state.model)(x)
    assert jnp.allclose(y_pred, y)

    import pprint

    pprint.pp(state)
