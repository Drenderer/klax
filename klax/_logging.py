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
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Generator, Iterable, Sequence
from time import time
from typing import Any

import equinox as eqx
import jax
from jaxtyping import Array, PyTree

from ._callbacks import Callback
from ._compat import HAS_TQDM, get_tqdm
from ._history import History
from ._losses import Loss
from ._trainstate import TrainingContext, TrainingState


class Metric(ABC):
    """Abstract metric.

    At its core, a metric is a simple function that computes a dictionary
    `dict[str, Array]` from a given [training state][klax.TrainingState].
    A metric additionally has a name that is used as a prefix to the
    dictionary keys. This avoids name conflicts, if the same keys are used
    in different metrics, e.g., `"training/loss"` and `"validation/loss"`
    for two metrics with names `"training"` and `"validation"`.

    Metrics are given to the [`MetricLogger`][klax.MetricLogger] callback,
    which evaluates the metrics periodically during training and adds the
    key value pairs to the [`History`][klax.History].

    !!!warning
        Stateful metrics can carry information about past evaluations and
        should therefore not be reused for different training runs.

    """

    name: str

    def __init__(self, name: str) -> None:
        """Initialize the metric.

        Args:
            name: The metrics namespace prefix.

        """
        self.name = name

    @abstractmethod
    def values(self, state: TrainingState) -> dict[str, Array]:
        """Compute the metric values.

        Args:
            state: The current [training state][klax.TrainingState] on which
                the metric is evaluated.

        Returns:
            A dictionary with pairs of metric names and values.

        """
        ...

    def __call__(self, state: TrainingState) -> dict[str, Array]:
        """Compute the metric values and prefix the key by the metrics name.

        Args:
            state: The current [training state][klax.TrainingState] on which
                the metric is evaluated.

        Returns:
            A dictionary with pairs of metric names prefixed by the metrics name
                and the corresponding values.

        """
        return {f"{self.name}/{k}": v for k, v in self.values(state).items()}


class LossMetric(Metric):
    """A metric based on a [`loss`][klax.Loss] and a dataset.

    A stateful metric, that owns it own batch generator to evaluate a loss
    function. The metric values are given by the loss' auxiliary quantities.

    !!!warning
        Reusing the same LossMetric instance for multiple training runs
        may result in different logged metric values due to the stateful nature
        of a batch generator. It is therefore recommended to create a new
        LossMetric instance for each run.
    """

    def __init__(
        self,
        loss: Loss,
        batch_generator: Generator[PyTree, None, None],
        name: str = "",
        vmap_ensemble: bool = False,
        jit_compile: bool = True,
    ):
        """Initialize the [`LossMetric`][klax.LossMetric].

        Args:
            loss: A [loss][klax.Loss] returning a scalar a dictionary of arrays
                values via the `aux` return.
            batch_generator: A `Generator` returning a batch of data on which
                the [loss][klax.Loss] is evaluated, every time the
                [LossMetric][klax.LossMetric] is called.
            name: A string prefix to prepend to the keys returned by loss.
            vmap_ensemble: Whether to vmap over the training state when
                evaluating the loss function. Set to `True` if the
                [`TrainingState`][klax.TrainingState] passed to
                [`LossMetric`][klax.LossMetric] contains an ensemble.
            jit_compile: Whether to jit compile the loss function for faster
                metric computations.

        """
        super().__init__(name=name)
        self.batch_generator = batch_generator
        func = lambda state, batch: loss(state.model, batch, state.run_state)
        if vmap_ensemble:
            func = eqx.filter_vmap(func, in_axes=(eqx.if_array(0), None))
        if jit_compile:
            func = eqx.filter_jit(func)
        self.func = func

    def values(self, state):
        batch = next(self.batch_generator)
        _, aux = self.func(state, batch)
        return aux


class MetricLogger(Callback):
    """Callback for logging [metrics][klax.Metric] in a [history][klax.History] during training."""

    def __init__(
        self,
        metrics: Sequence[Metric],
        log_every: int = 100,
    ):
        """Initialize the [`MetricLogger`][klax.MetricLogger].

        Args:
            metrics: Sequence of [metrics][klax.Metric] to evaluate and log.
            log_every: Frequency of logging metrics in steps. Defaults to 100.

        Raises:
            ValueError: If multiple metrics use the same name(-space).

        """
        metrics = list(metrics)
        names = [m.name for m in metrics]
        duplicates = [
            name for name, count in Counter(names).items() if count > 1
        ]
        if duplicates:
            raise ValueError(f"Duplicate metric namespaces: {duplicates}")

        self.metrics = metrics
        self.log_every = log_every

    def on_training_step(self, context: TrainingContext) -> None:
        if context.step % self.log_every == 0:
            for metric in self.metrics:
                aux = jax.device_get(metric(context.state))
                for key, value in aux.items():
                    context.history.append(key, context.step, value)

    def on_training_start(self, context: TrainingContext) -> None:
        self.start_time = time()
        self.steps_str_length = len(str(context.steps))
        self.on_training_step(context)

    def on_training_end(self, context: TrainingContext) -> None:
        end_time = time()
        context.history.total_time = end_time - self.start_time
        context.history.total_steps = context.step


class ProgressMeter(Callback):
    """A [callback][klax.Callback] for reporting the training progress."""

    make_progress_bar: bool
    update_every: int
    keys: set[str] | None
    exclude_keys: set[str] | None
    steps_str_length: int = 0
    tqdm_bar: Any = None

    def __init__(
        self,
        progress_bar: bool = True,
        update_every: int = 100,
        keys: Iterable[str] | None = None,
        exclude_keys: Iterable[str] | None = None,
    ):
        """Initialize a [`ProgressMeter`][klax.ProgressMeter].

        Args:
            progress_bar: Whether to print a progress bar. (Requires `tqdm`.)
            update_every: The step interval at which the progress meter shall
                be updated during training.
            keys: The keys of the metrics that shall be printed alongside the
                progress meter.
            exclude_keys: The keys of the metrics that shell be excluded from
                the progress meter. This parameter is ignored, if `keys` is not
                `None`.

        """
        if progress_bar and not HAS_TQDM:
            warnings.warn(
                "tqdm for progress bar not installed. "
                "Falling back to printing.",
                category=ImportWarning,
            )
            progress_bar = False
        self.make_progress_bar = progress_bar

        self.update_every = update_every

        if keys is not None and exclude_keys is not None:
            raise ValueError(
                "Specify either `keys` or `exclude_keys`, not both."
            )

        self.keys = set(keys) if keys is not None else None
        self.exclude_keys = (
            set(exclude_keys) if exclude_keys is not None else None
        )

    def _selected_keys(self, history: History) -> list[str]:
        all_keys = list(history.keys())

        if self.keys is not None:
            missing = self.keys - set(all_keys)
            if missing:
                warnings.warn(
                    f"Requested keys not found in history: {sorted(missing)}"
                )
            keys = [k for k in all_keys if k in self.keys]
        elif self.exclude_keys is not None:
            keys = [k for k in all_keys if k not in self.exclude_keys]
        else:
            keys = all_keys

        return sorted(keys)

    @staticmethod
    def _format_scalar(value: Any) -> str:
        return f"{float(value):.3e}"

    @classmethod
    def _format_value(cls, value: Any) -> str:
        if hasattr(value, "shape"):
            if value.shape == ():
                return cls._format_scalar(value)
            flat = value.reshape(-1)
            size = flat.shape[0]
            n_preview = min(3, size)
            preview = [cls._format_scalar(v) for v in flat[:n_preview].tolist()]
            suffix = ", ..." if size > n_preview else ""
            return "[" + ", ".join(preview) + suffix + "]"

        if isinstance(value, (list, tuple)):
            size = len(value)
            n_preview = min(3, size)
            preview = [cls._format_scalar(v) for v in value[:n_preview]]
            suffix = ", ..." if size > n_preview else ""
            return "[" + ", ".join(preview) + suffix + "]"

        return cls._format_scalar(value)

    def on_training_start(self, context: TrainingContext) -> None:
        # Guard against a leaked bar if start is somehow called twice.
        if self.tqdm_bar is not None:
            self.tqdm_bar.close()
            self.tqdm_bar = None

        self.steps_str_length = (
            len(str(context.steps)) if context.steps is not None else 0
        )

        if self.make_progress_bar:
            tqdm = get_tqdm()
            self.tqdm_bar = tqdm(
                total=context.steps,  # tqdm handles total=None fine (unbounded bar)
                initial=context.step,  # correct if resuming mid-run
                dynamic_ncols=True,
            )

    def on_training_step(self, context: TrainingContext) -> None:
        if context.step % self.update_every != 0:
            return

        keys = self._selected_keys(context.history)
        postfix = ", ".join(
            f"{key}={self._format_value(context.history[key].values[-1])}"
            for key in keys
        )

        if self.make_progress_bar and self.tqdm_bar is not None:
            self.tqdm_bar.set_postfix_str(postfix)
            # Set absolute position rather than incrementing, so the bar
            # can't drift out of sync with the real step count (e.g. on
            # resume, retries, or skipped steps).
            self.tqdm_bar.n = context.step
            self.tqdm_bar.refresh()
        else:
            step_str = (
                f"{context.step:>{self.steps_str_length}}/{context.steps}"
                if context.steps is not None
                else str(context.step)
            )
            print(f"Step {step_str}: {postfix}")

    def on_training_end(self, context: TrainingContext) -> None:
        if self.tqdm_bar is not None:
            # Make sure the bar reflects the true final step before closing.
            self.tqdm_bar.n = context.step
            self.tqdm_bar.refresh()
            self.tqdm_bar.close()
            self.tqdm_bar = None
