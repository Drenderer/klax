"""
Compare the klax.nn.MLP implementaiton with eqx.nn.MLP
"""

# %% Imports

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn.initializers import variance_scaling

from equinox.nn import MLP as EqxMLP

from klax.nn import MLP as KlaxMLP
from klax import fit, finalize
from klax import HistoryCallback

from matplotlib import pyplot as plt
from jaxtyping import Array


# Callback for recording predictions during training
class TrackPredictionHistory(HistoryCallback):
    x_eval: Array
    predictions: list
    pred_steps: list
    log_every_pred: int

    def __init__(
        self,
        x_eval: Array,
        log_every: int = 100,
        log_every_pred: int = 1000,
        verbose: bool = True,
    ):
        super().__init__(log_every=log_every, verbose=verbose)
        self.x_eval = x_eval
        self.predictions = []
        self.pred_steps = []
        self.log_every_pred = log_every_pred

    def __call__(self, cbargs):
        super().__call__(cbargs)
        if cbargs.step % self.log_every_pred == 0:
            model = finalize(cbargs.model)
            self.predictions.append(jax.vmap(model)(self.x_eval))
            self.pred_steps.append(cbargs.step)


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

klax_mlp = KlaxMLP(
    in_size,
    out_size,
    depth * [width],
    activation=activation,
    key=key1,
)
eqx_mlp = EqxMLP(
    in_size, out_size, width, depth, activation=activation, key=key2
)

# Train the models
klax_mlp, klax_hist = fit(
    klax_mlp,
    (x, y),
    steps=30_000,
    history=TrackPredictionHistory(x_eval, log_every_pred=log_every_pred),
    key=train_key,
)
eqx_mlp, eqx_hist = fit(
    eqx_mlp,
    (x, y),
    steps=30_000,
    history=TrackPredictionHistory(x_eval, log_every_pred=log_every_pred),
    key=train_key,
)

# %% Plot histories
ax = plt.subplot()
eqx_hist.plot(ax=ax, loss_options=dict(label="eqx MLP", c="blue"))
klax_hist.plot(ax=ax, loss_options=dict(label="klax MLP", c="orange"))
ax.set(
    yscale="log",
    ylabel="Loss",
    xlabel="Step",
)
ax.legend()
plt.show()

# %% Plot history of predictions
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))

y_klax = jnp.array(klax_hist.predictions)
y_eqx = jnp.array(eqx_hist.predictions)
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
    cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=klax_hist.steps[-1])
)
fig.colorbar(sm, ax=axes, orientation="vertical", label="Training Step")
plt.show()
