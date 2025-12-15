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
from typing import Any, Self

import jax
import optax
from jaxtyping import PyTree, PyTreeDef

from klax._losses import Loss


# TODO: Potentially rewrite this class as a non-dataclass, to get the __init__ function to work.
# I've tried, but got some weired error from jax on the filter_jit boundary in training_loop or made everything slower...
@jax.tree_util.register_dataclass
@dataclass
class TrainingState:
    """Dataclass of things that are expected to change during training.

    This dataclass combines the model parameters and the optimizer state into
    a single object that is passed around during training.
    Furthermore it implements the unflattening described in
    [low-overhead training loops][https://docs.kidger.site/equinox/tricks/]
    by exposing the model and optimizer state as properties which unflatten.
    This slightly reduces JAX's overhead when repeatedly passing through the
    jit boundary of the make_step function in the training loop.
    """

    model_leaves: list[Any]
    model_tree_def: PyTreeDef  # type: ignore
    opt_state_leaves: list[Any]
    opt_state_tree_def: PyTreeDef  # type: ignore

    @classmethod
    def create(cls, model: PyTree, opt_state: PyTree) -> Self:
        """Create a TrainingState from an unflattened model and optimizer state."""
        model_leaves, model_tree_def = jax.tree.flatten(model)
        opt_state_leaves, opt_state_tree_def = jax.tree.flatten(opt_state)
        return cls(
            model_leaves,
            model_tree_def,
            opt_state_leaves,
            opt_state_tree_def,
        )

    @property
    def model(self) -> PyTree:
        return jax.tree.unflatten(self.model_tree_def, self.model_leaves)

    @model.setter
    def model(self, value: PyTree) -> None:
        self.model_leaves, self.model_tree_def = jax.tree.flatten(value)

    @property
    def opt_state(self) -> PyTree:
        return jax.tree.unflatten(
            self.opt_state_tree_def, self.opt_state_leaves
        )

    @opt_state.setter
    def opt_state(self, value: PyTree) -> None:
        self.opt_state_leaves, self.opt_state_tree_def = jax.tree.flatten(
            value
        )


@dataclass
class TrainingStatic:
    """Dataclass of things that are expected to remain static during training."""

    # TODO: As of Python 3.13, PEP 712 (https://peps.python.org/pep-0712/) is not
    # yet implemented, so we cannot use the `converter` parameter. I also tried using
    # using an `equinox.Module` with `eqx.field` instead, but is messes with the initializer
    # input types, if there is a type conversion in the converter function.
    optimizer: optax.GradientTransformationExtraArgs
    batcher: Generator[PyTree[Any], None, None]
    batch_axes: PyTree[int | None]
    loss: Loss
    steps: int

    def __init__(
        self,
        optimizer: optax.GradientTransformation
        | optax.GradientTransformationExtraArgs,
        batcher: Generator[PyTree[Any], None, None],
        batch_axes: PyTree[int | None],
        loss: Loss,
        steps: int,
    ):
        self.optimizer = (
            optax.with_extra_args_support(optimizer)
            if not isinstance(optimizer, optax.GradientTransformationExtraArgs)
            else optimizer
        )
        self.batcher = batcher
        self.batch_axes = batch_axes
        self.loss = loss
        self.steps = steps
