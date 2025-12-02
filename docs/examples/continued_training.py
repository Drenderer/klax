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

"""Continued trainig example.

This example demonstrates how to continue a training session.
It shows how to initialize the optax initializer state with a previous training session's final state,
and how to use the `HistoryCallback` to add the training history to the subsequent training sessions.
"""

from functools import partial

import jax.random as jr
import optax
from matplotlib import pyplot as plt

import klax
from klax import HistoryCallback

key = jr.key(0)
data_key, model_key, train1_key, train2_key = jr.split(key, 4)

# Define data
x = jr.uniform(data_key, (1000, 2))
y = 2 * x.sum(axis=-1) + 1.0
y += 0.01 * jr.normal(data_key, y.shape)  # Add some noise


def history_factory():
    return HistoryCallback(
        metric_defs={
            "training_loss": ((x, y), partial(klax.mse, batch_axes=0)),
        },
        log_every=10,
        verbose=True,
    )


# A: Complete training for 2000 steps
model = klax.nn.MLP(2, "scalar", 2 * [16], key=model_key)
model, history_complete = klax.fit(
    model,
    (x, y),
    steps=2000,
    optimizer=optax.adabelief(1e-5),
    history=history_factory(),
    key=train1_key,
)

# B: Training split into two sessions with 1000 steps each.
#    The optimizer state of the second session is initialized
#    with the last optimizer state from the first session.
model = klax.nn.MLP(2, "scalar", 2 * [16], key=model_key)
model, history_continued = klax.fit(
    model,
    (x, y),
    steps=1000,
    optimizer=optax.adabelief(1e-5),
    history=history_factory(),
    key=train1_key,
)

model, history_continued = klax.fit(
    model,
    (x, y),
    steps=1000,
    optimizer=optax.adabelief(1e-5),  # (!) Same optimizer as in first session
    init_opt_state=history_continued.last_opt_state,  # Initialize the optimizer state with the last state
    history=history_continued,  # Continue the history
    key=train2_key,
)


# C: Training split into two sessions with reset optimizer state.
model = klax.nn.MLP(2, "scalar", 2 * [16], key=model_key)
model, history_reset = klax.fit(
    model,
    (x, y),
    steps=1000,
    optimizer=optax.adabelief(1e-5),
    history=history_factory(),
    key=train1_key,
)

model, history_reset = klax.fit(
    model,
    (x, y),
    steps=1000,
    optimizer=optax.adabelief(1e-5),  # (!) Same optimizer as in first session
    init_opt_state=None,  # No optimizer state is provided, so the optimizer is again initialized from scratch
    history=history_reset,  # Continue the history
    key=train2_key,
)


fig, ax = plt.subplots()
for histroy, label in zip(
    [history_complete, history_continued, history_reset],
    ["Continuous training", "Continued training", "Reset optimizer state"],
):
    ax.plot(
        history_complete.steps,
        history_complete.metrics["training_loss"],
        label=label,
    )
ax.set(
    title="Comparison of training loss histories",
    yscale="log",
    xlabel="Training steps",
    ylabel="Loss",
)
ax.legend()
plt.show()
