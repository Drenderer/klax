import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

import klax

x = jnp.linspace(0, 3, 1000)
y = jnp.sin(x)

data = (x, y)

key = jr.key(0)
keys = jr.split(key, 10)


@eqx.filter_vmap
def make_ensemble(key):
    return klax.nn.MLP("scalar", "scalar", [16, 16], key=key)


mlp_ensemble = make_ensemble(keys)

mlp_ensemble, history = klax.fit(
    mlp_ensemble,
    data,
    loss=klax.mse,
    steps=10_000,
    batch_size=32,
    vmap_ensemble=True,
    key=jr.key(0),
)

history.plot()
plt.show()
