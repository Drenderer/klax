from __future__ import annotations
from abc import ABC
from collections.abc import Callable
import datetime
import importlib
import time
from typing import Optional

import jax
from jaxtyping import PyTree, PyTreeDef, Scalar


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
    time_on_last_update: float
    data: PyTree
    val_data: PyTree | None
    _treedef_model: PyTreeDef
    _flat_model: list
    _treedef_opt_state: PyTreeDef
    _flat_opt_state: list
    _cache: dict = {}
    _get_loss: Callable[..., Scalar]
    _start_time: float

    def __init__(
        self,
        get_loss: Callable[[PyTree, PyTree], Scalar],
        treedef_model: PyTreeDef,
        treedef_opt_state: PyTreeDef,
        data: PyTree,
        val_data: Optional[PyTree] = None,
    ):
        self.data = data
        self.val_data = val_data
        self._get_loss = get_loss
        self._treedef_model = treedef_model
        self._treedef_opt_state = treedef_opt_state

    def update(self, flat_model: PyTree, flat_opt_state: PyTree, step: int):
        self._flat_model = flat_model
        self._flat_opt_state = flat_opt_state
        self.step = step
        self.time_on_last_update = time.time()

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
    def opt_state(self):
        return jax.tree_util.tree_unflatten(
            self._treedef_opt_state, self._flat_opt_state
        )

    @_lazy_evaluated_and_cached
    def loss(self):
        return self._get_loss(self.model, self.data)

    @_lazy_evaluated_and_cached
    def val_loss(self) -> Scalar | None:
        if self.val_data is None:
            return None
        return self._get_loss(self.model, self.val_data)


class Callback(ABC):
    """An abstract callback."""

    def __call__(self, cbargs: CallbackArgs) -> bool | None:
        """Called after each step during training."""
        pass

    def on_training_end(self, cbargs: CallbackArgs) -> None:
        """Called when training ends."""
        pass

    def on_training_start(self, cbargs: CallbackArgs) -> None:
        """Called when training starts."""
        pass


class HistoryCallback(Callback):
    log_every: int
    steps: list
    loss: list
    val_loss: list
    last_start_time: float  # start time of the last training
    last_end_time: float  # End time of the last training
    training_time: float = 0  # Total training time of all trainings
    verbose: bool
    step_offset: int = 0  # Potential offset due to previous trainings
    last_opt_state: PyTree | None = None

    def __init__(self, log_every: int = 100, verbose: bool = True):
        self.log_every = log_every
        self.verbose = verbose
        self.steps = []
        self.loss = []
        self.val_loss = []

    def __call__(self, cbargs: CallbackArgs):
        if cbargs.step % self.log_every == 0:
            self.steps.append(self.step_offset + cbargs.step)
            self.loss.append(cbargs.loss)
            self.val_loss.append(cbargs.val_loss)

            # Print message
            if self.verbose:
                message = f"Step: {cbargs.step}, Loss: {cbargs.loss:.3e}"
                if cbargs.val_data is not None:
                    message += f", Validation loss: {cbargs.val_loss:.3e}"
                print(message)

    def on_training_start(self, cbargs: CallbackArgs):
        self.last_start_time = cbargs.time_on_last_update
        if self.steps:
            self.step_offset = self.steps[-1]
        else:
            self(cbargs)

    def on_training_end(self, cbargs: CallbackArgs):
        self.last_end_time = cbargs.time_on_last_update
        self.training_time += self.last_end_time - self.last_start_time
        self.last_opt_state = cbargs.opt_state
        if self.verbose:
            print(f"Training took: {datetime.timedelta(seconds=self.training_time)}")

    def plot(self, *, ax=None, loss_options: dict = {}, val_loss_options: dict = {}):
        module_name = "matplotlib.pyplot"
        try:
            plt = importlib.import_module(module_name)
            if ax is None:
                _, ax = plt.subplots()
                ax.set(
                    xlabel="Step",
                    ylabel="Loss",
                    yscale="log",
                    title="Training History",
                )
                ax.grid(True)
            loss_options = dict(label="Loss", ls="-", c="black") | loss_options
            val_loss_options = (
                dict(label="Validation loss", ls="--", c="red") | val_loss_options
            )
            ax.plot(self.steps, self.loss, **loss_options)
            if any(x is not None for x in self.val_loss):
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
