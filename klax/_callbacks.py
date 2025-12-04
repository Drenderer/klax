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
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from jaxtyping import PyTree, Scalar

from ._trainstate import TrainingState, TrainingStatic


class Callback(ABC):
    """An abstract callback.

    Inherit from this class to create a custom callback.
    """

    def on_training_start(
        self, state: TrainingState, static: TrainingStatic
    ) -> None:
        """Call when training starts."""
        pass

    def __call__(
        self,
        state: TrainingState,
        step: int,
        step_loss: Scalar,
        static: TrainingStatic,
    ) -> bool | None:
        """Call after each step during training."""
        pass

    def on_training_end(
        self, state: TrainingState, static: TrainingStatic
    ) -> None:
        """Call when training ends."""
        pass


class HistoryCallback(Callback):
    """Default callback for logging a training process.

    Records loss histories, training time, and the last optimizer state.
    """

    log_every: int
    verbose: bool
    steps: list  #: List of steps at which the losses were recorded.
    metric_defs: dict[str, tuple[PyTree, Callable[[PyTree, PyTree], Scalar]]]
    metrics: dict[str, list[Scalar]]
    last_start_time: float  # start time of the last training
    last_end_time: float  # End time of the last training
    training_time: float = 0  # Total training time of all trainings
    step_offset: int = 0  # Potential offset due to previous trainings
    last_opt_state: PyTree | None = None

    def __init__(
        self, metric_defs={}, log_every: int = 100, verbose: bool = True
    ):
        """Initialize the `HistoryCallback`.

        Args:
            metric_defs: A dictionary defining the metrics to be recorded. Each key is
                the name of the metric, and each value is a tuple containing the data
                required to compute the metric and a callable that computes the metric
            log_every: Amount of steps after which the training and validation
                losses are logged. (Defaults to 100.)
            verbose: If true prints the training progress and losses.
                (Defaults to True.)

        """
        self.metric_defs = metric_defs
        self.log_every = log_every
        self.verbose = verbose
        self.steps = []
        self.metrics = defaultdict(list)
        self.total_steps_digits: int

    def __repr__(self):
        """Return a string representation of the HistoryCallback."""
        return (
            f"HistoryCallback(log_every={self.log_every}, "
            f"verbose={self.verbose})"
        )

    def on_training_start(self, state: TrainingState, static: TrainingStatic):
        """Initialize the training start time.

        Called at beginning of training.
        """
        self.last_start_time = time.time()
        self.total_steps_digits = len(str(static.steps))
        if self.steps:
            # If there are already steps, we assume that this is a continuation
            # of a training.
            self.step_offset = self.steps[-1]
        else:
            self(state, 0, jnp.array(jnp.nan), static)  # Log initial losses

    def __call__(
        self,
        state: TrainingState,
        step: int,
        step_loss: Scalar,
        static: TrainingStatic,
    ):
        """Record the losses and step count.

        Called at each step during training.
        """
        if step % self.log_every == 0:
            self.steps.append(self.step_offset + step)
            self.metrics["step_loss"].append(step_loss)
            for name, (data, metric_fn) in self.metric_defs.items():
                metric_value = metric_fn(state.model, data)
                self.metrics[name].append(metric_value)

            # Print message
            if self.verbose:
                print(
                    f"Step: {step:>{self.total_steps_digits}}: "
                    + ", ".join(
                        [
                            f"{name}: {self.metrics[name][-1]:.3e}"
                            for name in self.metrics.keys()
                        ]
                    )
                )

    def on_training_end(
        self, state: TrainingState, static: TrainingStatic
    ) -> None:
        """Record the training end time and the last optimizer state.

        Called at end of training.
        """
        self.last_end_time = time.time()
        last_training_time = self.last_end_time - self.last_start_time
        self.training_time += last_training_time
        self.last_opt_state = state.opt_state
        if self.verbose:
            print(
                f"Training took: {
                    datetime.timedelta(seconds=last_training_time)
                }"
            )

    def plot(
        self,
        *,
        ax: Any = None,
        names: list[str] | None = None,
    ):
        """Plot the recorded training and validation losses.

        Note:
            This method requires matplotlib.

        Args:
            ax: Matplotlib axes to plot into. If ``None`` then a new axis is
                created. (Defaults to None.)
            names: List of metric names to plot. If ``None``, all recorded
                metrics are plotted. (Defaults to None.)

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
                    ylabel="Metric",
                    yscale="log",
                    title="Training History",
                )
                ax.grid(True)

            if names is None:
                names = list(self.metrics.keys())
            for name in names:
                ax.plot(self.steps, self.metrics[name], label=name)

            ax.legend()
            return ax

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
