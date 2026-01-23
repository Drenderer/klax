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

from abc import ABC, abstractmethod
from typing import Any, Protocol

from jax import numpy as jnp
from jaxtyping import PyTree

from klax._callbacks import Callback
from klax._datahandler import BatchGenerator
from klax._losses import Loss


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
        batch_axis: Any,
        loss: Loss,
        num_batches: int = 1,
    ):
        self.batch = batcher(data, batch_size, batch_axis)
        self.batch_axis = batch_axis
        self.loss = loss
        self.num_batches = num_batches

    def __call__(self, model: PyTree) -> PyTree:
        losses = []
        for _ in range(self.num_batches):
            batch = next(self.batch)
            l = self.loss.value(model, batch, self.batch_axis)
            losses.append(l)
        return jnp.mean(jnp.stack(losses))


class History:
    """History container with some utility methods."""

    pass


class MetricLogger(Callback):
    """Callback for logging metrics in an History during training."""

    pass
