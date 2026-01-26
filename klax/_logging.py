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

"""Utilities for logging during training."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Any, Protocol

import jax
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from klax._callbacks import Callback
from klax._datahandler import BatchGenerator
from klax._losses import Loss
from klax._trainstate import TrainingView


class Metric(Protocol):
    """A metric that can be called on a model and returns a PyTree of results."""

    def __call__(self, model: PyTree) -> PyTree: ...


class LossMetric:
    """Compute loss over batches from a batcher."""

    def __init__(
        self,
        batcher: BatchGenerator,
        data: Any,
        batch_size: int,
        batch_axes: Any,
        loss: Loss,
        *,
        key: PRNGKeyArray,
    ):
        self.batch = batcher(data, batch_size, batch_axes, key=key)
        self.batch_axes = batch_axes
        self.loss = loss

    def __call__(self, model: PyTree) -> Array:
        batch = next(self.batch)
        return self.loss.value(model, batch, self.batch_axes)


type steps = list[int]
type values = list[Any]


class History:
    """History container with some utility methods."""

    content: dict[str, tuple[steps, values]]
    total_time: float  #: Total time spent in training
    total_steps: int  #: Total number of steps used in the training
    final_opt_state: PyTree  #: Final optimizer state after training

    def __init__(
        self,
        content: dict[str, tuple[steps, values]] | None = None,
        total_time: float = -1.0,
        total_steps: int = -1,
        final_opt_state: PyTree | None = None,
    ):
        self.content = content if content is not None else {}
        self.total_time = total_time
        self.total_steps = total_steps
        self.final_opt_state = final_opt_state

    def append(self, step: int, key: str, value: Any) -> None:
        if key not in self.content:
            self.content[key] = ([], [])
        self.content[key][0].append(step)
        self.content[key][1].append(value)

    def __getitem__(self, name: str) -> tuple[steps, values]:
        if name not in self.content:
            raise KeyError(f"Metric '{name}' not found in history.")
        return self.content[name]

    def save(self, path: str | Path) -> None:
        """Persist the history to disk using pickle.

        Args:
            path: Destination filepath where the history will be stored.

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "content": self.content,
            "total_time": self.total_time,
            "total_steps": self.total_steps,
            "final_opt_state": self.final_opt_state,
        }
        with path.open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str | Path) -> "History":
        """Restore a history saved with :meth:`save`.

        Args:
            path: Filepath to load the serialized history from.

        Returns:
            A populated History instance.

        """
        path = Path(path)
        with path.open("rb") as file:
            payload = pickle.load(file)

        return cls(
            content=payload.get("content", None),
            total_time=payload.get("total_time", -1.0),
            total_steps=payload.get("total_steps", -1),
            final_opt_state=payload.get("final_opt_state", None),
        )

    def plot(self, *keys: str, ax: Any = None, **kwargs) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "Failed to import matplotlib. Install it with: "
                "pip install klax[plotting]. "
                f"Original error: {str(e)}"
            )

        if ax is None:
            _, ax = plt.subplots()
            ax.set(
                xlabel="Step",
                ylabel="Metric",
                yscale="log",
                title="Training History",
            )
            ax.grid(True)
        keys = keys if keys else list(self.content.keys())
        for name in keys:
            steps, values = self.content[name]
            ax.plot(steps, values, label=name, **kwargs)
        ax.legend()
        return ax

    def extend(self, other: "History") -> None:
        """Extend this history with the contents of another history.

        Args:
            other: Another History instance to extend from.

        """
        for key, (other_steps, other_values) in other.content.items():
            if key not in self.content:
                self.content[key] = ([], [])
            self.content[key][0].extend(
                [s + self.total_steps for s in other_steps]
            )
            self.content[key][1].extend(other_values)

        self.total_time += other.total_time
        self.total_steps += other.total_steps
        self.final_opt_state = other.final_opt_state


class MetricLogger(Callback):
    """Callback for logging metrics in a History during training."""

    history: History
    log_every: int
    metric_defs: dict[str, (bool, Metric)]
    steps_str_length: int = 0
    verbose: bool = True
    start_time: float = 0.0

    def __init__(
        self,
        log_every: int = 100,
        metric_defs: dict[str, (bool, Metric)] | None = None,
        verbose: bool = True,
    ):
        """Initialize the MetricLogger.

        Args:
            log_every: Frequency of logging metrics (in steps).
            metric_defs: A dictionary mapping metric names to tuples of
                (whether to print the metric, metric function).
            verbose: Whether to print logged metrics to the console.

        """
        self.metric_defs = metric_defs or {}
        self.log_every = log_every
        self.history = History()
        self.verbose = verbose

    def add_metric(
        self, name: str, metric: Metric, verbose: bool = False
    ) -> None:
        """Add a metric to be logged during training.

        Args:
            name: Name of the metric.
            metric: A callable that computes the metric given the model.
            verbose: Whether to print the metric during logging.

        """
        self.metric_defs[name] = (verbose, metric)

    def on_training_step(self, view: TrainingView, step: int) -> None:
        """Log metrics at the current training step.

        Args:
            view: The current TrainingView containing state and static info.
            step: The current training step.

        """
        if step % self.log_every == 0:
            message = []
            for name, (verbose, metric_fn) in self.metric_defs.items():
                metric_value = jax.device_get(metric_fn(view.model))
                self.history.append(step, name, metric_value)
                if self.verbose and verbose:
                    try:
                        formatted_value = f"{metric_value:.4e}"
                    except TypeError:
                        formatted_value = str(metric_value)
                    message.append(f"{name}: {formatted_value}")

            if self.verbose:
                print(
                    f"Step {step:>{self.steps_str_length}}/{view.static.steps}: "
                    + ", ".join(message)
                )

    def on_training_start(self, view: TrainingView, step: int) -> None:
        self.start_time = time()
        self.steps_str_length = len(str(view.static.steps))
        self.on_training_step(view, step)

    def on_training_end(self, view: TrainingView, step: int) -> None:
        end_time = time()
        self.history.total_time = end_time - self.start_time
        self.history.total_steps = step
        self.history.final_opt_state = view.opt_state
