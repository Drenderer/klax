import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Self

import equinox as eqx
import jax
from jaxtyping import PyTree, Scalar

from ._losses import ValueFn
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
        self._step += 1

        # Clear cache
        self._cache.clear()


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

    @staticmethod
    @eqx.filter_jit
    def _loss_impl(value_fn: ValueFn, model: PyTree, batch: PyTree[Any]):
        return value_fn(model, batch)

    def loss(self, state: TrainingState) -> Scalar:
        self._ensure_step(state)
        if "loss" not in self._cache:
            self._cache["loss"] = self._loss_impl(
                self.value_fn, state.model, self.data
            )
        return self._cache["loss"]

    def val_loss(self, state: TrainingState) -> Scalar | None:
        self._ensure_step(state)
        if self.val_data is None:
            return None

        if "val_loss" not in self._cache:
            self._cache["val_loss"] = self._loss_impl(
                self.value_fn, state.model, self.val_data
            )
        return self._cache["val_loss"]


@dataclass
class TimingInfo:
    start_time: float | None = None
    total_time: float = 0.0
    time_of_last_update: float | None = None

    def update(self):
        time_of_last_update = time.time()
        if self.start_time is None:
            self.start_time = time_of_last_update
        else:
            self.total_time = time_of_last_update - self.start_time
        self.time_of_last_update = time_of_last_update


@dataclass
class TrainingContext:
    _state: TrainingState
    _evaluator: EvaluationContext
    _timer: TimingInfo

    def __init__(
        self,
        state: TrainingState,
        evaluator: EvaluationContext,
        timing: TimingInfo,
    ):
        self._state = state
        self._evaluator = evaluator
        self._timer = timing

    def update(self, flat_model: PyTree, flat_opt_state: PyTree) -> None:
        self._state.update(flat_model, flat_opt_state)
        self._timer.update()

    @property
    def flat_opt_state(self) -> PyTree:
        return self._state.flat_opt_state

    @property
    def flat_model(self) -> PyTree:
        return self._state.flat_model

    @property
    def model(self) -> PyTree:
        return self._state.model

    @property
    def treedef_model(self) -> PyTree:
        return self._state.treedef_model

    @property
    def opt_state(self) -> PyTree:
        return self._state.opt_state

    @property
    def treedef_opt_state(self) -> PyTree:
        return self._state.treedef_opt_state

    @property
    def step(self) -> int:
        return self._state.step

    @property
    def loss(self) -> Scalar:
        return self._evaluator.loss(self._state)

    @property
    def val_loss(self) -> Scalar | None:
        return self._evaluator.val_loss(self._state)

    @property
    def data(self) -> PyTree:
        return self._evaluator.data

    @property
    def val_data(self) -> PyTree:
        return self._evaluator.val_data

    @property
    def time_of_last_update(self) -> float | None:
        return self._timer.time_of_last_update

    @property
    def start_time(self) -> float | None:
        return self._timer.start_time

    @property
    def total_time(self) -> float | None:
        return self._timer.total_time
