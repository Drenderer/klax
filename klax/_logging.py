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
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from time import time
from typing import Any, Literal, Protocol, cast

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, PyTree

from ._callbacks import Callback
from ._compat import HAS_TQDM, get_plot, get_tqdm
from ._datahandler import Batcher
from ._trainstate import TrainingContext


class Metric(Protocol):
    """A Metric computes values that should be recorded in the training history.

    Metrics callables, that take the current [`TrainingContext`][klax.TrainingContext]
    and return some value to be added to the training history by the
    [`MetricLogger`][klax.MetricLogger].
    Additionally Metrics have a `name` and `verbose` property, that determines
    how they are logged.
    """

    name: str
    verbose: bool

    def __call__(self, context: TrainingContext) -> Any: ...


def metric[T](
    name: str, verbose: bool = False
) -> Callable[[Callable[[PyTree], T]], Callable[[PyTree], T]]:
    """Turn a function into a [Metric][klax.Metric] using a decorator factory.

    Intended usage:
    ```python
    @metric(name="my_metric", verbose=False)
    def compute_my_metric(context): ...
    ```

    Args:
        name: Name of the metric.
        verbose: Wether to print the Metrics values to the console during training.
            Defaults to False.

    """

    def make_metric(func):
        func.name = name
        func.verbose = verbose

        return cast(Metric, func)

    return make_metric


class BatchMetric:
    """Loss-like function [`Metric`][klax.Metric].

    A `BatchMetric` uses it's own batch generator and data, to turn a
    [loss][klax.Loss]-like function evaluation into a [`Metric`][klax.Metric].
    """

    def __init__[T](
        self,
        name: str,
        func: Callable[[PyTree, PyTree[Any, "T"], PyTree], Any],
        data: PyTree[Any, "T"],
        batcher: Batcher,
        batch_size: int,
        batch_axes: PyTree[int | str | None, "T ..."] = 0,
        verbose: bool = False,
        jit_compile: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the `BatchMetric`.

        Args:
            name: Name of the metric.
            func: The evaluation function to compute. It should take the model,
                a batch of data, and the auxiliary runtime state as input.
            data: The dataset to generate batches from.
            batcher: Batch generator factory.
            batch_size: The size of each batch.
            batch_axes: The axes corresponding to the batch dimension in the data.
            verbose: Verbosity level for the metric. Defaults to False.
            jit_compile: If true, `eqx.filter_jit` is used to jit compile `func`.
            key: PRNG key for random number generation.

        """
        self.name = name
        self.verbose = verbose
        self.batch_generator = batcher(data, batch_size, batch_axes, key=key)
        self.func = eqx.filter_jit(func) if jit_compile else func

    def __call__(self, context: TrainingContext) -> Any:
        """Compute the metric.

        Args:
            context: TrainingContext to evaluate in.

        Returns:
            The metric value on the sampled batch.

        """
        batch = next(self.batch_generator)
        return self.func(context.state.model, batch, context.state.run_state)


type Steps = list[int]
type Values = list[Any]


class History:
    """Dict-like object for storing a training history with metadata and utility methods.

    The training history stores (metric) values along with the
    training steps they correspond to, as well as total training
    time, total steps, and the final optimizer state.
    Furthermore, it provides methods for saving/loading the history
    to/from disk, plotting metrics, and extending the history.
    """

    content: dict[str, tuple[Steps, Values]]
    total_time: float  #: Total time spent in training
    total_steps: int  #: Total number of steps used in the training
    final_opt_state: PyTree  #: Final optimizer state after training

    def __init__(
        self,
        content: dict[str, tuple[Steps, Values]] | None = None,
        total_time: float = -1.0,
        total_steps: int = -1,
        final_opt_state: PyTree | None = None,
    ):
        self.content = content if content is not None else {}
        self.total_time = total_time
        self.total_steps = total_steps
        self.final_opt_state = final_opt_state

    def append(self, step: int, key: str, value: Any) -> None:
        """Add a new value to the history.

        Args:
            step: Training step the value belongs to.
            key: Metric name.
            value: Metric value.

        """
        if key not in self.content:
            self.content[key] = ([], [])
        self.content[key][0].append(step)
        self.content[key][1].append(value)

    def __getitem__(self, name: str) -> tuple[Steps, Values]:
        if name not in self.content:
            raise KeyError(f"Metric '{name}' not found in history.")
        return self.content[name]

    def keys(self) -> list[str]:
        """Get the list of metric names stored in the history.

        Returns:
            A list of metric names.

        """
        return list(self.content.keys())

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

    def plot(self, *keys: str, ax: Any = None, **kwargs: Any) -> None:
        """Plot stored metrics using matplotlib.

        Note:
            This method requires matplotlib.

        Args:
            keys: Metric names to plot. If empty, all metrics are plotted.
            ax: Matplotlib axes to plot into. If ``None`` then a new axis is
                created. (Defaults to None.)
            kwargs: Dictionary of keyword arguments passed to
                matplotlib's ``plot``.

        Raises:
            ImportError: If matplotlib is not installed.

        """
        plt = get_plot()

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
            kwargs["label"] = name  # Overwrite label if provided
            ax.plot(steps, values, **kwargs)
        ax.legend()
        return ax

    def extend(self, other: "History") -> None:
        """Extend this history with the contents of another history.

        Args:
            other: Another History instance to extend with.

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
    metrics: dict[str, Metric]
    steps_str_length: int = 0
    _verbose: Literal[0, 1, 2]
    start_time: float = 0.0
    progress_bar: bool
    tqdm_bar: Any = None

    def __init__(
        self,
        log_every: int = 100,
        metrics: Sequence[Metric] | None = None,
        verbose: Literal[0, 1, 2] = 2,
        history: History | None = None,
    ):
        """Initialize the MetricLogger.

        Args:
            log_every: Frequency of logging metrics (in steps).
            metrics: Sequence of [metrics][klax.Metric] to evaluate.
                If multiple metrics share the same name, the later metrics will
                overwrite prior metrics.
            verbose: Verbosity level for logging metrics to the console. If 0,
                no metrics will be printed. If 1, the metrics are printed. If
                2, a progress bar will be shown.
            history: An existing history object to log metrics to. If `None`,
                a new history object will be created.

        """
        self.metrics = {} if metrics is None else {m.name: m for m in metrics}
        self.log_every = log_every
        self.history = History() if history is None else history
        self._verbose = verbose
        if (verbose == 2) and not HAS_TQDM:
            warnings.warn(
                "tqdm for progress bar not installed. "
                "Falling back to verbosity level 1.",
                category=ImportWarning,
            )
            self._verbose = 1

    def add_metric(self, metric: Metric) -> None:
        """Add a metric to be logged during training.

        Warning:
            Existing metrics sharing the same name will be overwritten.

        Args:
            metric: The metric to be added.

        """
        self.metrics[metric.name] = metric

    def on_training_step(self, context: TrainingContext) -> None:
        """Log metrics at the current training step.

        Args:
            context: Current training context.

        """
        if context.step % self.log_every == 0:
            message = []
            for metric in self.metrics.values():
                metric_value = jax.device_get(metric(context))
                self.history.append(context.step, metric.name, metric_value)
                if self._verbose and metric.verbose:
                    try:
                        formatted_value = f"{metric_value:.4e}"
                    except TypeError:
                        formatted_value = str(metric_value)
                    message.append(f"{metric.name}: {formatted_value}")

            if self._verbose:
                postfix = ", ".join(message)
                if self._verbose > 1:
                    self.tqdm_bar.set_postfix_str(postfix)
                    if context.step != 0:
                        self.tqdm_bar.update(self.log_every)
                else:
                    print(
                        f"Step {context.step:>{self.steps_str_length}}/{context.steps}: "
                        + postfix
                    )

    def on_training_start(self, context: TrainingContext) -> None:
        self.start_time = time()
        self.steps_str_length = len(str(context.steps))

        if self._verbose > 1:
            tqdm = get_tqdm()
            self.tqdm_bar = tqdm(total=context.steps, dynamic_ncols=True)

        self.on_training_step(context)

    def on_training_end(self, context: TrainingContext) -> None:
        end_time = time()
        self.history.total_time = end_time - self.start_time
        self.history.total_steps = context.step
        self.history.final_opt_state = context.state.opt_state

        if self._verbose > 1:
            try:
                self.tqdm_bar.close()
            # TODO: Done make this a blanket exception
            except Exception:
                pass
