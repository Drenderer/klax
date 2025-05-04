"""
# Continued trainig example
This example demonstrates how to continue a training session.
It shows how to initialize the optax initializer state with a previous training session's final state,
and how to use the `HistoryCallback` to add the training history to the subsequent training sessions.
"""

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

ax = plt.gca()

# A: Complete training for 2000 steps
model = klax.nn.MLP(2, "scalar", 2 * [16], key=model_key)
model, history = klax.fit(
    model,
    (x, y),
    steps=2000,
    optimizer=optax.adabelief(1e-5),
    history=HistoryCallback(log_every=10, verbose=True),
    key=train1_key,
)

history.plot(ax=ax, loss_options=dict(label="Single session", ls="-", color="orange"))

# B: Training split into two sessions with 1000 steps each.
#    The optimizer state of the second session is initialized
#    with the last optimizer state from the first session.
model = klax.nn.MLP(2, "scalar", 2 * [16], key=model_key)
model, history = klax.fit(
    model,
    (x, y),
    steps=1000,
    optimizer=optax.adabelief(1e-5),
    history=HistoryCallback(log_every=10, verbose=True),
    key=train1_key,
)

last_opt_state = (
    history.last_opt_state
)  # Retrieve the last optimizer state from the history

model, history = klax.fit(
    model,
    (x, y),
    steps=1000,
    optimizer=optax.adabelief(1e-5),    # (!) Same optimizer as in first session
    init_opt_state=last_opt_state,      # Initialize the optimizer state with the last state
    history=history,                    # Continue the history
    key=train2_key,
)

history.plot(
    ax=ax,
    loss_options=dict(
        label="Split session, with init_opt_state", ls="--", color="black"
    ),
)


# C: Training split into two sessions with reset optimizer state.
model = klax.nn.MLP(2, "scalar", 2 * [16], key=model_key)
model, history = klax.fit(
    model,
    (x, y),
    steps=1000,
    optimizer=optax.adabelief(1e-5),
    history=HistoryCallback(log_every=10, verbose=True),
    key=train1_key,
)

model, history = klax.fit(
    model,
    (x, y),
    steps=1000,
    optimizer=optax.adabelief(1e-5),    # (!) Same optimizer as in first session
    init_opt_state=None,                # No optimizer state is provided, so the optimizer is again initialized from scratch
    history=history,                    # Continue the history
    key=train2_key,
)

history.plot(
    ax=ax,
    loss_options=dict(
        label="Split session, without init_opt_state", ls=":", color="black"
    ),
)


ax.set(yscale="log", title="Comparison of training loss histories")
ax.legend()
plt.show()
