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
from jax import numpy as jnp
from jaxtyping import Array, Int, PyTree, PyTreeDef

from klax._losses import Loss


# NOTE: Unfortunately step cannot be int, otherwise filter_jit does not trace it
@jax.tree_util.register_dataclass
@dataclass
class TrainingState:
    """Dataclass of things that are expected to change during training."""

    model: PyTree
    opt_state: PyTree
    aux: PyTree
    step: Int[Array, ""]


class TrainingContext:
    """Class of all training relevant objects."""

    _state: TrainingState | None
    _state_treedef: PyTreeDef  # type: ignore
    _state_leaves: list
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
        aux: PyTree,
        loss: Loss,
        steps: int,
    ):
        optimizer = (
            optax.with_extra_args_support(optimizer)
            if not isinstance(optimizer, optax.GradientTransformationExtraArgs)
            else optimizer
        )
        state = TrainingState(
            model, opt_state, aux, jnp.array(0, dtype=jnp.int32)
        )

        self._state = state
        self._state_leaves, self._state_treedef = jax.tree.flatten(state)
        self.optimizer = optimizer
        self.loss = loss
        self.batch_generator = batch_generator
        self.steps = steps

    @property
    def state(self) -> TrainingState:
        if self._state is None:
            state = jax.tree.unflatten(self._state_treedef, self._state_leaves)
            self._state = state

        return self._state

    @state.setter
    def state(self, value) -> None:
        if jax.tree.structure(value) != self._state_treedef:
            raise ValueError("PyTree strucutre of state changed.")
        self._state = value
        self._state_leaves, _ = jax.tree.flatten(value)

    def update_state(self, leaves):
        self._state = None
        self._state_leaves = leaves

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
    def aux(self) -> PyTree:
        return self.state.aux

    @aux.setter
    def aux(self, value) -> None:
        self.state.aux = value

    @property
    def step(self) -> int:
        return self.state.step
