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
from collections.abc import Generator, Iterable, Sequence
from time import time
from typing import Any, Literal, Protocol

import equinox as eqx
import jax
from jaxtyping import Array, PyTree

from ._callbacks import Callback
from ._compat import HAS_TQDM, get_tqdm
from ._history import History
from ._losses import Loss
from ._trainstate import TrainingContext, TrainingState


class Metric(Protocol):
    def __call__(self, state: TrainingState) -> dict[str, Array]: ...


class LossMetric:
    """LossMetric = Dataset + Loss."""

    def __init__(
        self,
        loss: Loss,
        batch_generator: Generator[PyTree, None, None],
        prefix: str = "",
        vmap_ensemble: bool = False,
        jit_compile: bool = True,
    ):
        self.batch_generator = batch_generator
        self.prefix = prefix
        loss_func = loss.value_and_aux
        if vmap_ensemble:
            loss_func = eqx.filter_vmap(
                loss_func, in_axes=(eqx.if_array(0), None, None)
            )
        if jit_compile:
            loss_func = eqx.filter_jit(loss_func)
        self.loss_func = loss_func

    def __call__(self, state: TrainingState) -> Any:
        batch = next(self.batch_generator)
        value, aux = self.loss_func(state.model, batch, state.run_state)
        if "loss" in aux:
            raise ValueError(
                f"aux from {type(self.loss_func).__name__} already contains "
                "'loss'; rename this component to avoid clashing with the "
                "auto-added total loss."
            )
        combined = {"loss": value, **aux}
        prefixed = {f"{self.prefix}/{k}": v for k, v in combined.items()}
        return prefixed


class MetricLogger(Callback):
    """Callback for logging metrics in a History during training."""

    def __init__(
        self,
        log_every: int = 100,
        metrics: Sequence[Metric] | None = None,
    ):
        """Initialize the MetricLogger.

        Args:
            log_every: Frequency of logging metrics (in steps).
            metrics: Sequence of [metrics][klax.Metric] to evaluate.
                If multiple metrics share the same name, the later metrics will
                overwrite prior metrics.

        """
        self.metrics = [] if metrics is None else list(metrics)
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
    """Callback for reporting the training progress."""

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
        all_keys = list(history.content.keys())

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
            preview = [
                cls._format_scalar(v) for v in flat[:n_preview].tolist()
            ]
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
