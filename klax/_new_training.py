import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import numpy as np
import optax
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from ._datahandler import BatchGenerator
from ._wrappers import apply, unwrap

# I have removed the idea of the datahandler class in favor of
# the generator style object that we used to use.
# Generating batches is best implemented imo as a generator.
# If someone ever needs to change the batch size or data during
# training, they can just create a new generator.


# Losses: I have implemented the losses in klax._losses

# Training State: I have implemented the basic training state, without any caching

# Training Loop: I have copied the training loop to _training

# Insight: My original idea was to put everything in the training state and
# let the callbacks modify the training state. However, this makes the training loop
# quite inefficient, since every change to the training state requires re-jitting the step function.

# TODO: Adapt callbacks to the new training state.
# TODO: Rewrite the fit function as outlined below


class Callback(ABC):
    """An abstract callback.

    Inherit from this class to create a custom callback.
    """

    def __call__(self, training_state: TrainingState) -> bool | None:
        """Call after each step during training."""
        pass

    def on_training_end(self, training_state: TrainingState) -> None:
        """Call when training ends."""
        pass

    def on_training_start(self, training_state: TrainingState) -> None:
        """Call when training starts."""
        pass


def fit(model, data, validation_data, loss_fn):
    # Initialize training state and callbacks
    loss = DefaultLoss(loss_fn)
    training_state = TrainingState(model, loss=loss)
    callbacks.append(history)
    training_state = training_loop(training_state, callbacks)
    return training_state.model, history
