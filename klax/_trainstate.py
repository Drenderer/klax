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
    """Dataclass of things that are expected to change during training.

    This consists of:

    * `model`: The `eqx.Module` (or more generally any PyTree)
        representing the trainable model.
    * `opt_state`: The state of the `optax` optimizer.
    * `run_state`: The user-defined state of the training run, which is
        passed to the loss function and may be modified via callbacks.
    * `step`: The current optimization step count of the training.
    """

    model: PyTree
    opt_state: PyTree
    run_state: PyTree
    step: Int[Array, ""]


class TrainingContext:
    """Collection of all training relevant objects.

    This includes:

    * `state`: The [`TrainingState`][klax.TrainingState]
    * `optimizer`: The optax optimizer
    * `loss`: The [Loss][klax.Loss] function
    * `batch_generator`: The generator object responsible for creating data
        batches
    * `steps`: The total number of scheduled optimization steps for the
        training run
    """

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
        run_state: PyTree,
        loss: Loss,
        steps: int,
    ):
        optimizer = (
            optax.with_extra_args_support(optimizer)
            if not isinstance(optimizer, optax.GradientTransformationExtraArgs)
            else optimizer
        )
        state = TrainingState(
            model, opt_state, run_state, jnp.array(0, dtype=jnp.int32)
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
