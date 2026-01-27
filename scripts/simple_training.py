import equinox as eqx
import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random as jr

import klax

key = jr.key(0)
model_key, train_key = jr.split(key, 2)

x = jnp.linspace(-3, 3, 100)
y = jnp.sin(x)

model = klax.nn.MLP("scalar", "scalar", [16], key=model_key)

get_bias = lambda m: m.layers[0].bias

logger = klax.MetricLogger(log_every=100)
logger.add_metric("parameter", lambda model: get_bias(model))


class MyCallback(klax.Callback):
    def on_training_step(self, view, step):
        if step % 1000 == 0:
            view.model = eqx.tree_at(
                get_bias, view.model, jnp.roll(get_bias(view.model), shift=1)
            )


model, history = klax.fit(
    model,
    (x, y),
    validation_data=(x, y),
    steps=30_000,
    logger=logger,
    # callbacks=[MyCallback()],
    key=train_key,
)

ax = history.plot("parameter")
ax.set(yscale="linear", title="Training History (Linear Scale)")
plt.show()

ax = history.plot("loss", "validation_loss")
plt.show()

y_pred = jax.vmap(model)(x)
fig, ax = plt.subplots()
ax.plot(x, y, label="True", marker="o", markevery=10)
ax.plot(x, y_pred, label="Predicted")
ax.legend()
plt.show()
