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

import jax.random as jr
import optax
from matplotlib import pyplot as plt

import klax

key = jr.key(0)
data_key, model_key, train1_key, train2_key = jr.split(key, 4)

# Define data
x = jr.uniform(data_key, (1000, 2))
y = 2 * x.sum(axis=-1) + 1.0
y += 0.01 * jr.normal(data_key, y.shape)  # Add some noise


def logger_factory():
    logger = klax.MetricLogger(log_every=10)
    logger.add_metric(
        "training_loss",
        klax.Evaluator(
            klax.mse,
            (x, y),
            klax.batch_data,
            batch_size=x.shape[0],
            batch_axes=0,
            key=data_key,
        ),
    )
    return logger


# A: Complete training for 2000 steps
model = klax.nn.MLP(2, "scalar", 2 * [16], key=model_key)
model, history_complete = klax.fit(
    model,
    (x, y),
    steps=2000,
    optimizer=optax.adabelief(1e-5),
    logger=logger_factory(),
    key=train1_key,
)

# B: Training split into two sessions with 100 and 1900 steps, respectively.
#    The optimizer state of the second session is initialized
#    with the last optimizer state from the first session.
#    We still expect slight differences from the continuous training,
#    since we get different batches in the second session.
model = klax.nn.MLP(2, "scalar", 2 * [16], key=model_key)
model, history_continued = klax.fit(
    model,
    (x, y),
    steps=100,
    optimizer=optax.adabelief(1e-5),
    logger=logger_factory(),
    key=train1_key,
)

model, history_continued_2 = klax.fit(
    model,
    (x, y),
    steps=1900,
    optimizer=optax.adabelief(1e-5),  # (!) Same optimizer as in first session
    init_opt_state=history_continued.final_opt_state,  # Initialize the optimizer state with the last state
    logger=logger_factory(),
    key=train2_key,
)
history_continued.extend(history_continued_2)

# C: Training split into two sessions with reset optimizer state.
model = klax.nn.MLP(2, "scalar", 2 * [16], key=model_key)
model, history_reset = klax.fit(
    model,
    (x, y),
    steps=100,
    optimizer=optax.adabelief(1e-5),
    logger=logger_factory(),
    key=train1_key,
)

model, history_reset_2 = klax.fit(
    model,
    (x, y),
    steps=1900,
    optimizer=optax.adabelief(1e-5),  # (!) Same optimizer as in first session
    init_opt_state=None,  # No optimizer state is provided, so the optimizer is again initialized from scratch
    logger=logger_factory(),
    key=train2_key,
)
history_reset.extend(history_reset_2)

# D: Plot the recorded losses
fig, ax = plt.subplots()
legend_labels = []
for histroy, label in zip(
    [history_complete, history_continued, history_reset],
    ["Continuous training", "Continued training", "Reset optimizer state"],
):
    histroy.plot("training_loss", ax=ax)
    legend_labels.append("Loss - " + label)
ax.legend(legend_labels)
ax.set(
    title="Comparison of training loss histories",
    yscale="log",
    xlabel="Training steps",
    ylabel="Loss",
)
plt.show()
