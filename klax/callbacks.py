from __future__ import annotations
from collections.abc import Callable
import importlib
import typing
from typing import Optional, Protocol

import jax
from jaxtyping import PyTree, PyTreeDef, Scalar

from .typing import DataTree


class CallbackArgs:
    """
    Callback arguments object, designed to work in conjunction with ``klax.fit``.

    It should not be used elsewhere!

    The instances of this class are passed to any callback object in the fit
    function. The class implements cached and lazy-evaluated values via property
    methods. This means that properties like training_loss are only calculated
    if they are used and are stored such that they are not calculated multiple
    times.
    """

    step: int
    data: DataTree
    val_data: DataTree | None
    _treedef_model: PyTreeDef
    _flat_model: list
    _cache: dict = {}
    _get_loss: Callable[..., Scalar]

    def __init__(
        self,
        get_loss: Callable[[PyTree, DataTree], Scalar],
        treedef_model: PyTreeDef,
        data: DataTree,
        val_data: Optional[DataTree] = None,
    ):
        self.data = data
        self.val_data = val_data
        self._get_loss = get_loss
        self._treedef_model = treedef_model

    def update(self, flat_model: PyTree, step: int):
        self._flat_model = flat_model
        self.step = step

        # Clear cache
        self._cache = {}

    @staticmethod
    def _lazy_evaluated_and_cached(fun: Callable) -> property:
        """Turns a public method into a property

        The return value of `fun`is stored in the `_cache` dictionary of the
        current object using the function name as key. If the name is already in
        `_cache` then the cached value is simply returned, wihout evaluating
        `fun`.

        Args:
            fun: Method to wrap.

        Returns:
            Wraped method as a property.
        """
        attr_name = fun.__name__

        def wrapper(self):
            if attr_name not in self._cache:
                self._cache.setdefault(attr_name, fun(self))
            return self._cache.get(attr_name)

        return property(wrapper)

    @_lazy_evaluated_and_cached
    def model(self):
        return jax.tree_util.tree_unflatten(self._treedef_model, self._flat_model)

    @_lazy_evaluated_and_cached
    def loss(self):
        return self._get_loss(self.model, self.data)

    @_lazy_evaluated_and_cached
    def val_loss(self) -> Scalar | None:
        if self.val_data is None:
            return None
        return self._get_loss(self.model, self.val_data)


@typing.runtime_checkable
class Callback(Protocol):
    """An abstract callback."""

    def __call__(self, cbargs: CallbackArgs) -> bool | None:
        raise NotImplementedError


@typing.runtime_checkable
class HistoryCallback(Protocol):
    """An abstract history callback."""

    def __init__(self, log_every: int) -> None:
        raise NotImplementedError

    def __call__(self, cbargs: CallbackArgs) -> bool | None:
        raise NotImplementedError

    def add_training_time(self, seconds: float) -> None:
        raise NotImplementedError


class DefaultHistoryCallback:
    log_every: int
    steps: list
    loss: list
    val_loss: list
    training_time: float

    def __init__(self, log_every: int = 100):
        self.log_every = log_every
        self.steps = []
        self.loss = []
        self.val_loss = []
        self.training_time = 0.0

    def __call__(self, cbargs: CallbackArgs):
        if cbargs.step % self.log_every == 0:
            self.steps.append(cbargs.step)
            self.loss.append(cbargs.loss)
            if cbargs.val_loss is not None:
                self.val_loss.append(cbargs.val_loss)

    def add_training_time(self, seconds: float):
        self.training_time += seconds

    def plot(
        self,
        *,
        ax=None,
        loss_options: dict = {},
        val_loss_options: dict = {},
    ):
        module_name = "matplotlib.pyplot"
        try:
            plt = importlib.import_module(module_name)
            if ax is None:
                _, ax = plt.subplots()
            loss_options = dict(label="Loss", ls="-", c="black") | loss_options
            val_loss_options = (
                dict(label="Validation loss", ls="--", c="red") | val_loss_options
            )
            ax.plot(self.steps, self.loss, **loss_options)
            if self.val_loss is not None:
                ax.plot(self.steps, self.val_loss, **val_loss_options)
        except ImportError as e:
            raise ImportError(
                f"Failed to import module '{module_name}'. "
                f"Install it with: pip install klax[plotting]. "
                f"Original error: {str(e)}"
            )

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
