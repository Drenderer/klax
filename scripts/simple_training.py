import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random as jr

import klax

key = jr.key(0)
model_key, train_key = jr.split(key, 2)

x = jnp.linspace(-1, 1, 100)
y = jnp.sin(x)

model = klax.nn.MLP("scalar", "scalar", [16], key=model_key)

logger = klax.MetricLogger()
logger.add_metric("parameter", lambda model: model.layers[0].weight[0])

model, history = klax.fit(
    model,
    (x, y),
    validation_data=(x, y),
    steps=10_000,
    logger=logger,
    key=train_key,
)

ax = history.plot("parameter", color="black")
ax.set(yscale="linear", title="Training History (Linear Scale)")
plt.show()

ax = history.plot("loss", "validation_loss")
plt.show()
