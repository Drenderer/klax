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

"""Reduce on plateau example.

This example demonstrates how to use the `reduce_on_plateau` function from `optax` to
reduce the learning rate when the training loss plateaus. A custom callback is used to
track the learning rate scale during training.
"""

import equinox as eqx
import jax
import jax.random as jr
import optax
from matplotlib import pyplot as plt
from optax import contrib
from optax import tree_utils as otu

import klax

key = jr.key(0)

# Define data
key, data_key = jr.split(key, 2)
x = jr.uniform(data_key, (100, 1))
y = 2.0 * x**2 + 1.0

# Define model
model_key, training_key = jr.split(key, 2)
model = eqx.nn.MLP(1, 1, 16, 2, activation=jax.nn.softplus, key=model_key)

# Define optimizer
opt = optax.chain(
    optax.adam(1e-2),
    contrib.reduce_on_plateau(
        patience=5,
        cooldown=10,
        factor=0.5,
        accumulation_size=200,
    ),
)


class TrackScale(klax.Callback):
    scales: list
    steps: list
    log_every: int

    def __init__(self, log_every=100):
        self.scales = []
        self.steps = []
        self.log_every = log_every

    def on_training_step(self, view, step):
        if step % self.log_every == 0:
            scale = otu.tree_get(view.opt_state, "scale")
            self.scales.append(scale)
            self.steps.append(step)


scale_tracker = TrackScale(log_every=100)

# Train
model, hist = klax.fit(
    model,
    (x, y),
    steps=30000,
    optimizer=opt,
    callbacks=[scale_tracker],
    key=training_key,
)

# Plot training loss and scale
plt.figure(figsize=(8, 4))
ax = plt.subplot(1, 2, 1)
hist.plot(ax=ax)
ax.set(
    title="Training Loss",
    yscale="log",
)
ax.grid(True)

ax = plt.subplot(1, 2, 2)
ax.plot(scale_tracker.steps, scale_tracker.scales, label="Scale")
ax.set(title="Learning rate scale", yscale="log")
ax.grid(True)

plt.tight_layout()
plt.show()
