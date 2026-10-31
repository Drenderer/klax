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


@klax.loss
def my_loss(model, batch, run_state):
    x, y = batch
    y_pred = jax.vmap(model)(x)
    diff = y - y_pred
    mse = jnp.mean(diff**2)
    mae = jnp.mean(jnp.abs(diff))
    return mse, {"mae": mae, "mse": mse}


def get_bias(state):
    return {"bias": state.model.layers[0].bias}


model, history = klax.fit(
    model,
    (x, y),
    validation_data=(x, y),
    steps=30_000,
    batcher=klax.batch_data,
    loss=my_loss,
    metrics=[get_bias],
    key=train_key,
)

history.plot(exclude_keys=["bias"])
plt.show()

fig, ax = plt.subplots()
history.plot("bias", ax=ax)
ax.set(yscale="linear")
plt.show()

y_pred = jax.vmap(model)(x)
fig, ax = plt.subplots()
ax.plot(x, y, label="True", marker="o", markevery=10)
ax.plot(x, y_pred, label="Predicted")
ax.legend()
plt.show()
