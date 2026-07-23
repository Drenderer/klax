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

from collections import namedtuple
from collections.abc import Generator

import jax
import optax
from jaxtyping import PyTree

from ._losses import Loss

_TrainingState = namedtuple(
    "_TrainingState", ["model", "opt_state", "run_state"]
)


class TrainingState:
    """PyTree of the training state, i.e., model, optimizer state and run state.

    This PyTree largely behaves like a named tuple with attributes `model`,
    `opt_state` and `run_state`.
    Internally, however, it implements the unflattening trick, mentioned by
    Patrick Kidger [here](https://docs.kidger.site/equinox/tricks/#low-overhead-training-loops).
    This means that instead of storing the PyTree of the named tuple internally,
    the flattened version is stored. Only when a specific attribute, e.g.,
    `state.model` is used, the flattened internal representation is unflattened.
    This yields slight performance benefits in tight training loops, where
    the state is repeatedly passed to a jitted function performing a single
    gradient update step. Since when crossing the jit-boundary, JAX would
    flatten and unflatten the model, we can save the overhead of flattening, by
    storing the flat PyTree outside of jit.
    To avoid unnecessary unflattening, the unflattend PyTree is cached. When
    crossing jit boundaries the cache is emptied.

    This class essentially behaves just like
    ```python
        @jax.tree_util.register_dataclass
        @dataclass
        class TrainingState:
            model: PyTree
            opt_state: PyTree
            run_state: PyTree

            def replace(self, *, model=None, opt_state=None, run_state=None):
                return TrainingState(
                    model if model is not None else self.model,
                    opt_state if opt_state is not None else self.opt_state,
                    run_state if run_state is not None else self.run_state,
                )
    ```

    Raises:
        AttributeError: When trying to modify the attributes.

    """

    __slots__ = ("_leaves", "_treedef", "_cache")

    def __init__(
        self, model: PyTree, opt_state: PyTree, run_state: PyTree
    ) -> None:
        tree = _TrainingState(model, opt_state, run_state)
        leaves, treedef = jax.tree.flatten(tree)
        self._leaves = leaves
        self._treedef = treedef
        self._cache = None

    def _get(self):
        if self._cache is None:
            self._cache = jax.tree_util.tree_unflatten(
                self._treedef, self._leaves
            )
        return self._cache

    @property
    def model(self):
        return self._get().model

    @model.setter
    def model(self, value):
        raise AttributeError(
            "TrainingState is immutable; use .replace(model=...) instead."
        )

    @property
    def opt_state(self):
        return self._get().opt_state

    @opt_state.setter
    def opt_state(self, value):
        raise AttributeError(
            "TrainingState is immutable; use .replace(opt_state=...) instead."
        )

    @property
    def run_state(self):
        return self._get().run_state

    @run_state.setter
    def run_state(self, value):
        raise AttributeError(
            "TrainingState is immutable; use .replace(run_state=...) instead."
        )

    def replace(
        self, *, model=None, opt_state=None, run_state=None
    ) -> "TrainingState":
        current = self._get()
        return TrainingState(
            model if model is not None else current.model,
            opt_state if opt_state is not None else current.opt_state,
            run_state if run_state is not None else current.run_state,
        )

    @staticmethod
    def _flatten(state):
        return state._leaves, state._treedef

    @staticmethod
    def _unflatten(treedef, leaves):
        obj = TrainingState.__new__(TrainingState)
        obj._leaves = leaves
        obj._treedef = treedef
        obj._cache = None
        return obj


jax.tree_util.register_pytree_node(
    TrainingState, TrainingState._flatten, TrainingState._unflatten
)


class TrainingContext:
    """Collection of all training relevant objects.

    This includes:

    * `state`: The [`TrainingState`][klax.TrainingState].
    * `optimizer`: The optax optimizer.
    * `loss`: The [Loss][klax.Loss] function.
    * `batch_generator`: The generator object responsible for creating data
        batches.
    * `step`: The number of currently completed training steps.
    * `steps`: The total number of scheduled optimization steps for the
        training run.
    """

    state: TrainingState
    optimizer: optax.GradientTransformationExtraArgs
    loss: Loss
    batch_generator: Generator[PyTree, None, None]
    step: int
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
        state = TrainingState(model, opt_state, run_state)

        self.state = state
        self.optimizer = optimizer
        self.loss = loss
        self.batch_generator = batch_generator
        self.step = 0
        self.steps = steps

    def update(self, state, step):
        self.state = state
        self.step = step
