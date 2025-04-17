"""
Compare the klax.nn.MLP implementaiton with eqx.nn.MLP
"""

import jax
from klax.nn import MLP as KlaxMLP
from klax import fit
from equinox.nn import MLP as EqxMLP

import jax.numpy as jnp
import jax.random as jr
from matplotlib import pyplot as plt

# Define a simple dataset
key = jr.key(0)
x = jnp.linspace(-2, 2, 100)[:, None]
y = jax.vmap(jnp.inner)(x, x) 
y += 0.3 * jr.normal(key, shape=y.shape)

# Define model parameters
in_size = x.shape[-1]
out_size = "scalar"
width = 16
depth = 2
hidden_layers = 2
activation = jax.nn.softplus

# Initialize models
key1, key2, train_key = jax.random.split(key, 3)
klax_mlp = KlaxMLP(in_size, out_size, depth*[width], activation=activation, key=key1)
eqx_mlp = EqxMLP(in_size, out_size, width, depth, activation=activation, key=key2)

# Train the models
klax_mlp, klax_hist = fit(klax_mlp, (x, y), steps=5_000, key=train_key)
eqx_mlp, eqx_hist = fit(eqx_mlp, (x, y), steps=5_000, key=train_key)

# Plot histories
ax = plt.subplot()
klax_hist.plot(ax=ax, loss_options=dict(label='klax MLP', c='orange'))
eqx_hist.plot(ax=ax, loss_options=dict(label='eqx MLP'))
ax.legend()
plt.show()

