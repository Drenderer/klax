"""Compare the klax.nn.MLP implementaiton with eqx.nn.MLP."""

# %% Imports

import jax
import jax.numpy as jnp
import jax.random as jr
from equinox.nn import MLP as EQXMLP
from jax.nn.initializers import variance_scaling
from jaxtyping import Array
from matplotlib import pyplot as plt

from klax import Metric, TrainingState, finalize, fit
from klax.nn import MLP as KLAXMLP


# Metric for recording predictions during training
class TrackPredictionMetric(Metric):
    x_eval: Array

    def __init__(self, x_eval: Array, name: str = "predictions"):
        super().__init__(name)
        self.x_eval = x_eval

    def values(self, state: TrainingState) -> dict[str, Array]:
        model = finalize(state.model)
        return {"y": jax.vmap(model)(self.x_eval)}


# Define a simple dataset
key = jr.key(0)
x = jnp.linspace(-2, 2, 20)
x_eval = jnp.linspace(-3, 3, 100)
y = x**2
y += 0.3 * jr.normal(key, shape=y.shape)

# Define model parameters
in_size = "scalar"
out_size = "scalar"
width = 16
depth = 2
hidden_layers = 2
activation = jax.nn.softplus

log_every = 1000

# %% Initialize models
key1, key2, train_key = jax.random.split(key, 3)

# Weight init from equinox
eqx_w_init = variance_scaling(
    scale=1 / 3, mode="fan_in", distribution="uniform"
)

klax_mlp = KLAXMLP(
    in_size,
    out_size,
    depth * [width],
    activation=activation,
    key=key1,
)
eqx_mlp = EQXMLP(
    in_size, out_size, width, depth, activation=activation, key=key2
)

# Train the models
klax_mlp, klax_hist = fit(
    klax_mlp,
    (x, y),
    steps=30_000,
    metrics=[TrackPredictionMetric(x_eval)],
    log_every=log_every,
    key=train_key,
)

eqx_mlp, eqx_hist = fit(
    eqx_mlp,
    (x, y),
    steps=30_000,
    metrics=[TrackPredictionMetric(x_eval)],
    log_every=log_every,
    key=train_key,
)

# %% Plot histories
ax = plt.subplot()
ax.plot(*eqx_hist["training/mse"], label="eqx MLP", c="blue")
ax.plot(*klax_hist["training/mse"], label="klax MLP", c="orange")
ax.set(
    yscale="log",
    ylabel="Loss",
    xlabel="Step",
)
ax.legend()
plt.show()

# %% Plot history of predictions
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))

steps, y_klax = klax_hist["predictions/y"]
steps, y_eqx = eqx_hist["predictions/y"]
y_klax = jnp.array(y_klax)
y_eqx = jnp.array(y_eqx)
colors = plt.cm.jet(jnp.linspace(0, 1, y_klax.shape[0]))

axes[0].scatter(x, y, label="Data", marker="x", c="black")
axes[0].set(
    xlabel="x",
    ylabel="y",
    title="Klax",
)
axes[1].scatter(x, y, label="Data", marker="x", c="black")
axes[1].set(
    xlabel="x",
    title="Equinox",
)

for yk, ye, c in zip(y_klax, y_eqx, colors):
    axes[0].plot(x_eval, yk, c=c)
    axes[1].plot(x_eval, ye, c=c)

sm = plt.cm.ScalarMappable(
    cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=steps[-1])
)
fig.colorbar(sm, ax=axes, orientation="vertical", label="Training Step")
plt.show()
