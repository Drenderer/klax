"""Compare the klax.nn.MLP implementaiton with eqx.nn.MLP."""

# %% Imports

import jax
import jax.numpy as jnp
import jax.random as jr
from equinox.nn import MLP as EQXMLP
from jax.nn.initializers import variance_scaling
from jaxtyping import Array
from matplotlib import pyplot as plt

from klax import MetricLogger, finalize, fit
from klax.nn import MLP as KLAXMLP


# Callback for recording predictions during training
class TrackPredictionMetric:
    x_eval: Array

    def __init__(self, x_eval: Array):
        self.x_eval = x_eval

    def __call__(self, model):
        model = finalize(model)
        return jax.vmap(model)(self.x_eval)


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

log_every_pred = 1000

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
logger = MetricLogger()
logger.add_metric("predictions", TrackPredictionMetric(x_eval))
klax_mlp, klax_hist = fit(
    klax_mlp,
    (x, y),
    steps=30_000,
    logger=logger,
    key=train_key,
)

logger = MetricLogger()
logger.add_metric("predictions", TrackPredictionMetric(x_eval))
eqx_mlp, eqx_hist = fit(
    eqx_mlp,
    (x, y),
    steps=30_000,
    logger=logger,
    key=train_key,
)

# %% Plot histories
ax = plt.subplot()
eqx_hist.plot("loss", ax=ax, label="eqx MLP", c="blue")
klax_hist.plot("loss", ax=ax, label="klax MLP", c="orange")
ax.set(
    yscale="log",
    ylabel="Loss",
    xlabel="Step",
)
ax.legend()
plt.show()

# %% Plot history of predictions
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))

steps, y_klax = klax_hist["predictions"]
steps, y_eqx = eqx_hist["predictions"]
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
