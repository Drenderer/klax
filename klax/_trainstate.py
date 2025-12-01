from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import optax
from jaxtyping import PyTree, PyTreeDef

from klax._losses import Loss


@jax.tree_util.register_dataclass
@dataclass
class TrainingState:
    model: PyTree
    opt_state: PyTree


# @jax.tree_util.register_pytree_node_class
# class TrainingState:
#     model_leaves: list[Any]
#     model_tree_def: PyTreeDef # type: ignore
#     opt_state_leaves: list[Any]
#     opt_state_tree_def: PyTreeDef # type: ignore


#     def __init__(self, model: PyTree, opt_state: PyTree) -> None:
#         self.model_leaves, self.model_tree_def = jax.tree.flatten(model)
#         self.opt_state_leaves, self.opt_state_tree_def = jax.tree.flatten(opt_state)

#     @property
#     def model(self) -> PyTree:
#         return jax.tree.unflatten(self.model_tree_def, self.model_leaves)

#     @model.setter
#     def model(self, value: PyTree) -> None:
#         self.model_leaves, self.model_tree_def = jax.tree.flatten(value)

#     @property
#     def opt_state(self) -> PyTree:
#         return jax.tree.unflatten(self.opt_state_tree_def, self.opt_state_leaves)

#     @opt_state.setter
#     def opt_state(self, value: PyTree) -> None:
#         self.opt_state_leaves, self.opt_state_tree_def = jax.tree.flatten(value)

#     def tree_flatten(self):
#         return (
#             (self.model_leaves, self.opt_state_leaves),
#             (self.model_tree_def, self.opt_state_tree_def),
#         )

#     @classmethod
#     def tree_unflatten(
#         cls,
#         aux_data: tuple[PyTreeDef, PyTreeDef], # type: ignore
#         children: list[Any],
#     ) -> "TrainingState":
#         model_tree_def, opt_state_tree_def = aux_data
#         model_leaves, opt_state_leaves = children
#         model = jax.tree.unflatten(model_tree_def, model_leaves)
#         opt_state = jax.tree.unflatten(opt_state_tree_def, opt_state_leaves)
#         return cls(model, opt_state)


@dataclass
class TrainingStatic:
    optimizer: optax.GradientTransformation
    batcher: Iterable[PyTree[Any]]
    batch_axes: PyTree[int | None]
    loss: Loss
    steps: int
