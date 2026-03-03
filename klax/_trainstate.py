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

from collections.abc import Generator
from dataclasses import dataclass

import jax
import optax
from jaxtyping import PyTree

from klax._losses import Loss


@jax.tree_util.register_dataclass
@dataclass
class TrainingState:
    """Dataclass of things that are expected to change during training."""

    model: PyTree
    opt_state: PyTree
    aux_state: PyTree
    step: int


class TrainingContext:
    """Dataclass of things that are expected to remain static during training."""

    state: TrainingState
    optimizer: optax.GradientTransformationExtraArgs
    loss: Loss
    batch_generator: Generator[PyTree, None, None]
    steps: int

    def __init__(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation
        | optax.GradientTransformationExtraArgs,
        opt_state: PyTree,
        batch_generator: Generator[PyTree, None, None],
        aux_state: PyTree,
        loss: Loss,
        steps: int,
    ):
        optimizer = (
            optax.with_extra_args_support(optimizer)
            if not isinstance(optimizer, optax.GradientTransformationExtraArgs)
            else optimizer
        )

        self.state = TrainingState(model, opt_state, aux_state, 0)
        self.optimizer = optimizer
        self.loss = loss
        self.batch_generator = batch_generator
        self.steps = steps

    @property
    def model(self) -> PyTree:
        return self.state.model

    @model.setter
    def model(self, value) -> None:
        self.state.model = value

    @property
    def opt_state(self) -> PyTree:
        return self.state.opt_state

    @opt_state.setter
    def opt_state(self, value) -> None:
        self.state.opt_state = value

    @property
    def aux_state(self) -> PyTree:
        return self.state.aux_state

    @aux_state.setter
    def aux_state(self, value) -> None:
        self.state.aux_state = value

    @property
    def step(self) -> int:
        return self.state.step
