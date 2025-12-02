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

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import optax
from jaxtyping import PyTree, PyTreeDef

from klax._losses import Loss


# TODO: Potentially rewrite this class as a non-dataclass, to get the __init__ function to work. I've tried, but got some weired error from jax on the filter_jit boundary in training_loop.
# @jax.tree_util.register_pytree_node_class
@jax.tree_util.register_dataclass
@dataclass
class TrainingState:
    model_leaves: list[Any]
    model_tree_def: PyTreeDef  # type: ignore
    opt_state_leaves: list[Any]
    opt_state_tree_def: PyTreeDef  # type: ignore

    @classmethod
    def create(cls, model: PyTree, opt_state: PyTree) -> "TrainingState":
        model_leaves, model_tree_def = jax.tree.flatten(model)
        opt_state_leaves, opt_state_tree_def = jax.tree.flatten(opt_state)
        return cls(
            model_leaves,
            model_tree_def,
            opt_state_leaves,
            opt_state_tree_def,
        )

    # def __init__(self, model: PyTree, opt_state: PyTree) -> None:
    #     self.model_leaves, self.model_tree_def = jax.tree.flatten(model)
    #     self.opt_state_leaves, self.opt_state_tree_def = jax.tree.flatten(opt_state)

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

    # def tree_flatten(self):
    #     return (
    #         (self.model_leaves, self.opt_state_leaves),
    #         (self.model_tree_def, self.opt_state_tree_def),
    #     )

    # @classmethod
    # def tree_unflatten(
    #     cls,
    #     aux_data: tuple[PyTreeDef, PyTreeDef],  # type: ignore
    #     children: tuple,
    # ) -> "TrainingState":
    #     model_tree_def, opt_state_tree_def = aux_data
    #     model_leaves, opt_state_leaves = children
    #     model = jax.tree.unflatten(model_tree_def, model_leaves)
    #     opt_state = jax.tree.unflatten(opt_state_tree_def, opt_state_leaves)
    #     return cls(model, opt_state)


@dataclass
class TrainingStatic:
    optimizer: optax.GradientTransformation
    batcher: Iterable[PyTree[Any]]
    batch_axes: PyTree[int | None]
    loss: Loss
    steps: int
