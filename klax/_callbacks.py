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


from abc import ABC

from ._trainstate import TrainingContext


class Callback(ABC):
    """Callback base class.

    A callback consists of three methods:
        - `on_training_start`: Executed once at the start of training.
        - `on_training_step`: Executed after each step (parameter update)
            during training.
        - `on_training_end`: Executed once at the end of training.

    Each method receives the current step and a [`TrainingView`][klax.TrainingView]
    object that provides read and write access to the current
    training state (model and optimizer state) as well as read-only
    access to static training information (loss function, optimizer,
    batch axes, etc). The `on_training_step` method can optionally
    return a boolean "stop signal", that - if `True` - will stop the
    training at the current step.

    Inherit from this class and overwrite one or more methods to
    create a custom callback.
    """

    def on_training_start(self, context: TrainingContext, step: int) -> None:
        """Execute at the beginning of training, before any parameter updates."""
        pass

    def on_training_step(
        self, context: TrainingContext, step: int
    ) -> bool | None:
        """Execute after each parameter update during training."""
        pass

    def on_training_end(self, context: TrainingContext, step: int) -> None:
        """Execute at the end of training."""
        pass
