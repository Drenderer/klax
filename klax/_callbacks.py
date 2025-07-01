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

import datetime
import importlib
import pickle
import time
from abc import ABC
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import jax
from jaxtyping import PyTree, PyTreeDef, Scalar


class CallbackArgs:
    """A callback argument designed to work in conjunction with [`klax.fit`][].

    This class should not be instantiated directly. An instance of this class
    is passed to every callback object in the fit function. When writing a
    custom callback, use the properties of this class to access the current
    model, optimizer state, training data, and validation data during training.

    This class implements cached and lazy-evaluated values via property
    methods. This means that properties like ``loss`` are only calculated if
    they are used and are stored such that they are not calculated multiple
    times.
    """

    step: int  #: Current step-count of the training.
    time_on_last_update: float  #: Global time of the last :meth:`update` call.
    data: PyTree  #: PyTree of the training data.
    val_data: PyTree | None  #: PyTree of the validation data.
    _treedef_model: PyTreeDef
    _flat_model: list
    _treedef_opt_state: PyTreeDef
    _flat_opt_state: list
    _cache: dict = {}
    _get_loss: Callable[[PyTree, PyTree], Scalar]
    _start_time: float

    def __init__(
        self,
        get_loss: Callable[[PyTree, PyTree], Scalar],
        treedef_model: PyTreeDef,
        treedef_opt_state: PyTreeDef,
        data: PyTree,
        val_data: PyTree | None = None,
    ):
        """Initialize the callback arguments object.

        Args:
        get_loss: Function that takes a model and a batch of data and
            returns the loss.
        treedef_model: Tree structure of the model.
        treedef_opt_state: Tree structure of the :py:mod:`optax` optimizer.
        data: PyTree of the training data.
        val_data: PyTree of the validation data. If None, no validation
            loss is calculated and the property :py:attr:`val_loss` will
            return None.

        """
        self.data = data
        self.val_data = val_data
        self._get_loss = get_loss
        self._treedef_model = treedef_model
        self._treedef_opt_state = treedef_opt_state

    def update(self, flat_model: PyTree, flat_opt_state: PyTree, step: int):
        """Update the callback arguments object with the current model and optimizer state.

        This method is called repeatedly in [`klax.fit`][].

        Args:
            flat_model: Flattened PyTree of the model.
            flat_opt_state: Flattened PyTree of the `optax`
                optimizer.
            step: Current step-count of the training.

        """
        self._flat_model = flat_model
        self._flat_opt_state = flat_opt_state
        self.step = step
        self.time_on_last_update = time.time()

        # Clear cache
        self._cache = {}

    @staticmethod
    def _lazy_evaluated_and_cached(fun: Callable[[Any], Any]) -> property:
        """Turn a public method into a property.

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

    @_lazy_evaluated_and_cached
    def model(self):
        """Lazy-evaluated and cached model."""
        return jax.tree_util.tree_unflatten(
            self._treedef_model, self._flat_model
        )

    @_lazy_evaluated_and_cached
    def opt_state(self):
        """Lazy-evaluated and cached optimizer state."""
        return jax.tree_util.tree_unflatten(
            self._treedef_opt_state, self._flat_opt_state
        )

    @_lazy_evaluated_and_cached
    def loss(self):
        """Lazy-evaluated and cached training loss."""
        return self._get_loss(self.model, self.data)

    @_lazy_evaluated_and_cached
    def val_loss(self) -> Scalar | None:
        """Lazy-evaluated and cached validation loss."""
        if self.val_data is None:
            return None
        return self._get_loss(self.model, self.val_data)


class Callback(ABC):
    """An abstract callback.

    Inherit from this class to create a custom callback.
    """

    def __call__(self, cbargs: CallbackArgs) -> bool | None:
        """Call after each step during training."""
        pass

    def on_training_end(self, cbargs: CallbackArgs) -> None:
        """Call when training ends."""
        pass

    def on_training_start(self, cbargs: CallbackArgs) -> None:
        """Call when training starts."""
        pass


class HistoryCallback(Callback):
    """Default callback for logging a training process.

    Records loss histories, training time, and the last optimizer state.
    """

    log_every: int
    steps: list #: List of steps at which the losses were recorded.
    loss: list
    val_loss: list
    last_start_time: float  # start time of the last training
    last_end_time: float  # End time of the last training
    training_time: float = 0  # Total training time of all trainings
    verbose: bool
    step_offset: int = 0  # Potential offset due to previous trainings
    last_opt_state: PyTree | None = None

    def __init__(self, log_every: int = 100, verbose: bool = True):
        """Initialize the `HistoryCallback`.

        Args:
        log_every: Amount of steps after which the training and validation
            losses are logged. (Defaults to 100.)
        verbose: If true prints the training progress and losses.
            (Defaults to True.)

        """
        self.log_every = log_every
        self.verbose = verbose
        self.steps = []
        self.loss = []
        self.val_loss = []

    def __repr__(self):
        """Return a string representation of the HistoryCallback."""
        return (
            f"HistoryCallback(log_every={self.log_every}, "
            f"verbose={self.verbose})"
        )

    def __call__(self, cbargs: CallbackArgs):
        """Record the losses and step count.

        Called at each step during training.
        """
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
        """Initialize the training start time.

        Called at beginning of training.
        """
        self.last_start_time = cbargs.time_on_last_update
        if self.steps:
            # If there are already steps, we assume that this is a continuation
            # of a training.
            self.step_offset = self.steps[-1]
        else:
            self(cbargs)

    def on_training_end(self, cbargs: CallbackArgs):
        """Record the training end time and the last optimizer state.

        Called at end of training.
        """
        self.last_end_time = cbargs.time_on_last_update
        self.training_time += self.last_end_time - self.last_start_time
        self.last_opt_state = cbargs.opt_state
        if self.verbose:
            print(
                f"Training took: {datetime.timedelta(seconds=self.training_time)}"
            )

    def plot(
        self, *, ax=None, loss_options: dict = {}, val_loss_options: dict = {}
    ):
        """Plot the recorded training and validation losses.

        Note:
            This method requires matplotlib.

        Args:
            ax: Matplotlib axes to plot into. If ``None`` then a new axis is
                created. (Defaults to None.)
            loss_options: Dictionary of keyword arguments passed to
                matplotlibs ``plot`` for the training loss. (Defaults to {}.)
            val_loss_options: Dictionary of keyword arguments passed to
                matplotlibs ``plot`` for the validation loss. (Defaults to {}.)

        Raises:
            ImportError: _description_

        """
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

            # Rename the color option to "c", if it exists. Otherwise "c" and
            # "color" are both passed to plot, which causes an error.
            if "color" in loss_options:
                loss_options["c"] = loss_options.pop("color")
            if "color" in val_loss_options:
                val_loss_options["c"] = val_loss_options.pop("color")

            loss_options = dict(label="Loss", ls="-", c="black") | loss_options
            val_loss_options = (
                dict(label="Validation loss", ls="--", c="red")
                | val_loss_options
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

    def save(
        self,
        filename: str | Path,
        overwrite: bool = False,
        create_dir: bool = True,
    ) -> None:
        """Save the HistoryCallback instance to a file using pickle.

        Args:
            filename: The file path where the instance should be saved.
            overwrite: If True, overwrite the file if it already exists.
                If False, raise a FileExistsError if the file exists.
                (Defaults to False.)
            create_dir: If True, create the parent directory if it does not
                exist. (Defaults to True.)

        Raises:
            FileExistsError: If the file already exists and overwrite is False.
            ValueError: If the provided path is not a valid file path.

        """
        filename = Path(filename)

        if filename.suffix == "":
            filename = filename.with_suffix(".pkl")
        assert filename.suffix == ".pkl", "File must have a .pkl suffix."

        if filename.exists() and not overwrite:
            raise FileExistsError(
                f"The file '{filename}' already exists. Use overwrite=True to "
                f"overwrite it."
            )

        if create_dir:
            filename.parent.mkdir(parents=True, exist_ok=True)

        with filename.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str | Path) -> "HistoryCallback":
        """Load a `HistoryCallback` instance from a file.

        Args:
            filename: The file path from which the instance should be loaded.

        Returns:
            The loaded `HistoryCallback` instance.

        Raises:
            ValueError: If the file is not a valid pickle file or does not
                contain a `HistoryCallback` instance.

        """
        filename = Path(filename)

        with filename.open("rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, HistoryCallback):
            raise ValueError(
                f"The file '{filename}' does not contain a valid "
                f"HistoryCallback instance."
            )

        return obj
