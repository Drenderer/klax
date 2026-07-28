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

import warnings
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from time import time
from typing import Any, Literal, Protocol, cast

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ._callbacks import Callback
from ._compat import HAS_TQDM, get_tqdm
from ._datahandler import Batcher
from ._history import History
from ._trainstate import TrainingContext, TrainingState


class Metric(Protocol):
    def __call__(self, state: TrainingState) -> Array: ...


class BatchedMetric:
    """Loss-like function [`Metric`][klax.Metric].

    A `BatchMetric` uses it's own batch generator and data, to turn a
    [loss][klax.Loss]-like function evaluation into a [`Metric`][klax.Metric].
    """

    def __init__[T](
        self,
        loss_func: Callable[[PyTree, PyTree[Any, "T"], PyTree], Any],
        batch_generator: Generator[PyTree, None, None],
    ):
        self.loss_func = loss_func
        self.batch_generator = batch_generator

    def __call__(self, state: TrainingState) -> Any:
        batch = next(self.batch_generator)
        return self.loss_func(state.model, batch, state.run_state)


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

        if self._verbose > 1:
            try:
                self.tqdm_bar.close()
            # TODO: Done make this a blanket exception
            except Exception:
                pass
