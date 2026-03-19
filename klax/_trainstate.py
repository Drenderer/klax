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
from typing import Any

import jax
import optax
from jaxtyping import PyTree, PyTreeDef

from ._losses import Loss

# ====--------------------------------------------------------------------=== #
# TrainingView classes
# ====--------------------------------------------------------------------=== #


@jax.tree_util.register_dataclass
@dataclass
class TrainingState:
    """Dataclass of things that are expected to change during training.

    This dataclass combines the model parameters and the optimizer state into
    a single object that is passed around during training.
    """

    model_leaves: list
    opt_state_leaves: list


@dataclass(frozen=True)
class TrainingStatic:
    """Dataclass of things that are expected to remain static during training.

    It provides an interface to unflatten and flatten the model and optimizer
    from their flat states.
    """

    # TODO: As of Python 3.13, PEP 712 (https://peps.python.org/pep-0712/) is not
    # yet implemented, so we cannot use the `converter` parameter. I also tried using
    # using an `equinox.Module` with `eqx.field` instead, but is messes with the initializer
    # input types, if there is a type conversion in the converter function.
    model_tree_def: PyTreeDef  # pyright: ignore[reportInvalidTypeForm]
    optimizer: optax.GradientTransformationExtraArgs
    opt_state_tree_def: PyTreeDef  # pyright: ignore[reportInvalidTypeForm]
    batch: Generator[PyTree[Any], None, None]
    batch_axes: PyTree[int | None]
    loss: Loss
    steps: int

    def assemble_model(self, model_leaves):
        return jax.tree.unflatten(self.model_tree_def, model_leaves)

    def disassemble_model(self, model):
        leaves, treedef = jax.tree.flatten(model)
        if treedef != self.model_tree_def:
            raise ValueError("Model structure changed")
        return leaves

    def assemble_opt_state(self, opt_state_leaves):
        return jax.tree.unflatten(self.opt_state_tree_def, opt_state_leaves)

    def disassemble_opt_state(self, opt_state):
        leaves, treedef = jax.tree.flatten(opt_state)
        if treedef != self.opt_state_tree_def:
            raise ValueError("Opt state structure changed")
        return leaves


# This is similar to the old CallbackArgs, but ensures a clean separation
# between mutable state (TrainingState) and static components (TrainingStatic),
# while providing a nice public-facing interface to access and modify the model
# and optimizer state.
class TrainingView:
    """Interface for accessing the training state and static components.

    Provides properties to access and modify the model and optimizer state.

    Attributes:
        static: The immutable [training static][klax.TrainingStatic].
        model: The model instance.
        opt_state: The optimizer state.

    """

    _state: TrainingState
    _static: TrainingStatic
    _model: Any  # Cached model instance.
    _opt_state: Any  # Cached optimizer state.

    def __init__(self, state: TrainingState, static: TrainingStatic):
        self._state = state
        self._static = static
        self._model = None
        self._opt_state = None

    @property
    def model(self):
        """Accessor for the model instance."""
        if self._model is None:
            self._model = self._static.assemble_model(self._state.model_leaves)
        return self._model

    @model.setter
    def model(self, value):
        self._state.model_leaves = self._static.disassemble_model(value)
        self._model = value

    @property
    def opt_state(self):
        """Accessor for the optimizer state."""
        if self._opt_state is None:
            self._opt_state = self._static.assemble_opt_state(
                self._state.opt_state_leaves
            )
        return self._opt_state

    @opt_state.setter
    def opt_state(self, value):
        self._state.opt_state_leaves = self._static.disassemble_opt_state(
            value
        )
        self._opt_state = value

    @property
    def model_tree_def(self) -> PyTreeDef:  # pyright: ignore[reportInvalidTypeForm]
        return self._static.model_tree_def

    @property
    def optimizer(self) -> optax.GradientTransformationExtraArgs:
        return self._static.optimizer

    @property
    def opt_state_tree_def(self) -> PyTreeDef:  # pyright: ignore[reportInvalidTypeForm]
        return self._static.opt_state_tree_def

    @property
    def batch(self) -> Generator[PyTree[Any], None, None]:
        return self._static.batch

    @property
    def batch_axes(self) -> PyTree[int | None]:
        return self._static.batch_axes

    @property
    def loss(self) -> Loss:
        return self._static.loss

    @property
    def steps(self) -> int:
        return self._static.steps


# ====--------------------------------------------------------------------=== #
# Factory methods
# ====--------------------------------------------------------------------=== #


def make_view(
    model: PyTree[Any],
    optimizer: optax.GradientTransformation
    | optax.GradientTransformationExtraArgs,
    opt_state: PyTree[Any],
    batch: Generator[PyTree[Any], None, None],
    batch_axes: PyTree[int | None],
    loss: Loss,
    steps: int,
) -> TrainingView:
    """Create the TrainingView from the model and optimizer.

    Args:
        model: The initial model parameters.
        optimizer: The optimizer to use for training.
        opt_state: The initial optimizer state.
        batch: A generator that yields batches of data.
        batch_axes: A PyTree indicating the batch axes for each component of the data.
        loss: The loss function to use for training.
        steps: The total number of training steps.

    Returns:
        A TrainingView.

    """
    model_leaves, model_treedef = jax.tree.flatten(model)
    opt_state_leaves, opt_state_treedef = jax.tree.flatten(opt_state)

    optimizer = (
        optax.with_extra_args_support(optimizer)
        if not isinstance(optimizer, optax.GradientTransformationExtraArgs)
        else optimizer
    )

    state = TrainingState(
        model_leaves=model_leaves,
        opt_state_leaves=opt_state_leaves,
    )

    static = TrainingStatic(
        model_tree_def=model_treedef,
        optimizer=optimizer,
        opt_state_tree_def=opt_state_treedef,
        batch=batch,
        batch_axes=batch_axes,
        loss=loss,
        steps=steps,
    )

    return TrainingView(state, static)
