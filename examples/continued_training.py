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

"""Continued training example.

This example demonstrates how to continue a training session.
It shows how to capture the final optimizer state of a training session with a
`klax.Callback`, how to initialize the optimizer state of a subsequent session
with it via `init_opt_state`, and how to add the history of a subsequent
session to the previous one with `klax.History.append`.
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


def metrics_factory():
    # LossMetrics own a stateful batch generator, so we create new ones for
    # every training session.
    metric = klax.LossMetric(
        klax.mse,
        klax.batch_data((x, y), x.shape[0], 0, key=data_key),
        name="full_batch",
    )
    return [
        metric,
    ]


class KeepOptState(klax.Callback):
    """Store the optimizer state at the end of a training session."""

    opt_state = None

    def on_training_end(self, context: klax.TrainingContext) -> None:
        self.opt_state = context.state.opt_state


def extend(history: klax.History, other: klax.History) -> None:
    """Append the entries of a subsequent session's history to `history`.

    The steps of `other` are offset by the number of steps in `history`.
    The initial (step 0) entries of `other` are skipped, since they coincide
    with the final entries of `history`.
    """
    offset = history.total_steps
    for key in other.keys():
        for step, value in zip(*other[key]):
            if step > 0:
                history.append(key, offset + step, value)
    history.total_steps += other.total_steps
    history.total_time += other.total_time


# A: Complete training for 2000 steps
model = klax.nn.MLP(2, "scalar", 2 * [16], key=model_key)
model, history_complete = klax.fit(
    model,
    (x, y),
    steps=2000,
    optimizer=optax.adabelief(1e-5),
    log_every=10,
    metrics=metrics_factory(),
    key=train1_key,
)

# B: Training split into two sessions with 100 and 1900 steps, respectively.
#    The optimizer state of the second session is initialized
#    with the last optimizer state from the first session.
#    We still expect slight differences from the continuous training,
#    since we get different batches in the second session.
model = klax.nn.MLP(2, "scalar", 2 * [16], key=model_key)
keep_opt_state = KeepOptState()
model, history_continued = klax.fit(
    model,
    (x, y),
    steps=100,
    optimizer=optax.adabelief(1e-5),
    log_every=10,
    metrics=metrics_factory(),
    callbacks=[keep_opt_state],
    key=train1_key,
)

model, history_continued_2 = klax.fit(
    model,
    (x, y),
    steps=1900,
    optimizer=optax.adabelief(1e-5),  # (!) Same optimizer as in first session
    init_opt_state=keep_opt_state.opt_state,  # Initialize the optimizer state with the last state
    log_every=10,
    metrics=metrics_factory(),
    key=train2_key,
)
extend(history_continued, history_continued_2)

# C: Training split into two sessions with reset optimizer state.
model = klax.nn.MLP(2, "scalar", 2 * [16], key=model_key)
model, history_reset = klax.fit(
    model,
    (x, y),
    steps=100,
    optimizer=optax.adabelief(1e-5),
    log_every=10,
    metrics=metrics_factory(),
    key=train1_key,
)

model, history_reset_2 = klax.fit(
    model,
    (x, y),
    steps=1900,
    optimizer=optax.adabelief(1e-5),  # (!) Same optimizer as in first session
    init_opt_state=None,  # No optimizer state is provided, so the optimizer is again initialized from scratch
    log_every=10,
    metrics=metrics_factory(),
    key=train2_key,
)
extend(history_reset, history_reset_2)

# D: Plot the recorded losses
fig, ax = plt.subplots()
legend_labels = []
for history, label in zip(
    [history_complete, history_continued, history_reset],
    ["Continuous training", "Continued training", "Reset optimizer state"],
):
    history.plot("full_batch/mse", ax=ax)
    legend_labels.append("Loss - " + label)
ax.legend(legend_labels)
ax.set(
    title="Comparison of training loss histories",
    yscale="log",
    xlabel="Training steps",
    ylabel="Loss",
)
plt.show()
