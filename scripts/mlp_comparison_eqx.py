"""
Compare the klax.nn.MLP implementaiton with eqx.nn.MLP
"""

import jax
from equinox.nn import MLP as EqxMLP

from klax.nn import MLP as KlaxMLP
from klax import fit, unwrap
from klax.callbacks import HistoryCallback

import jax.numpy as jnp
import jax.random as jr
from matplotlib import pyplot as plt
from jaxtyping import Array


# Callback for recording predictions during training
class TrackPredictionHistory(HistoryCallback):
    x_eval: Array
    predictions: list

    def __init__(self, x_eval: Array, log_every: int = 100, verbose: bool = True):
        super().__init__(log_every=log_every, verbose=verbose)
        self.x_eval = x_eval
        self.predictions = []

    def __call__(self, cbargs):
        super().__call__(cbargs)
        if cbargs.step % self.log_every == 0:
            model = unwrap(cbargs.model)
            self.predictions.append(jax.vmap(model)(self.x_eval))


# Define a simple dataset
key = jr.key(0)
x = jnp.linspace(-2, 2, 20)
x_eval = jnp.linspace(-2, 3, 100)
y = x**2
y += 0.3 * jr.normal(key, shape=y.shape)

# Define model parameters
in_size = "scalar"
out_size = "scalar"
width = 16
depth = 2
hidden_layers = 2
activation = jax.nn.softplus

log_every = 500

# %% Initialize models
key1, key2, train_key = jax.random.split(key, 3)
klax_mlp = KlaxMLP(in_size, out_size, depth * [width], activation=activation, key=key1)
eqx_mlp = EqxMLP(in_size, out_size, width, depth, activation=activation, key=key2)

# Train the models
klax_mlp, klax_hist = fit(
    klax_mlp,
    (x, y),
    steps=30_000,
    history=TrackPredictionHistory(x_eval, log_every=log_every),
    key=train_key,
)
eqx_mlp, eqx_hist = fit(
    eqx_mlp, 
    (x, y), 
    steps=30_000, 
    history=TrackPredictionHistory(x_eval, log_every=log_every), 
    key=train_key
)

# %% Plot histories
ax = plt.subplot()
eqx_hist.plot(ax=ax, loss_options=dict(label="eqx MLP", c="blue"))
klax_hist.plot(ax=ax, loss_options=dict(label="klax MLP", c="orange"))
ax.set(
    yscale='log',
    ylabel = 'Loss',
    xlabel='Step',
)
ax.legend()
plt.show()

# %% Plot predictions
# ax = plt.subplot()
# ax.scatter(x, y, label="Data", marker="x", c="black")
# ax.plot(x_eval, jax.vmap(unwrap(eqx_mlp))(x_eval), ls="-", c="blue", label="eqx.MLP")
# ax.plot(
#     x_eval, jax.vmap(unwrap(klax_mlp))(x_eval), ls="-.", c="orange", label="klax.MLP"
# )
# ax.legend()
# plt.show()

# %% Plot history of predictions
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

y_klax = jnp.array(klax_hist.predictions)
y_eqx = jnp.array(eqx_hist.predictions)
colors = plt.cm.jet(jnp.linspace(0,1,y_klax.shape[0]))

axes[0].scatter(x, y, label="Data", marker="x", c="black")
axes[1].scatter(x, y, label="Data", marker="x", c="black")

for yk, ye, c in zip(y_klax, y_eqx, colors):
    axes[0].plot(x_eval, yk, c=c)
    axes[1].plot(x_eval, ye, c=c)

plt.show()
