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


@klax.metric("bias")
def get_bias(context):
    return context.state.model.layers[0].bias


@klax.loss
def my_loss(model, batch, run_state):
    x, y = batch
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y - y_pred) ** 2)


model, history = klax.fit(
    model,
    (x, y),
    validation_data=(x, y),
    steps=30_000,
    metrics=[get_bias],
    # callbacks=[MyCallback()],
    batcher=klax.batch_data,
    loss=my_loss,
    key=train_key,
)

steps, values = history["bias"]
fig, ax = plt.subplots()
ax.plot(steps, values)
ax.set(xlabel="step", yscale="linear", title="Recorded bias values")
plt.show()

ax = history.plot("loss", "validation_loss")
plt.show()

y_pred = jax.vmap(model)(x)
fig, ax = plt.subplots()
ax.plot(x, y, label="True", marker="o", markevery=10)
ax.plot(x, y_pred, label="Predicted")
ax.legend()
plt.show()
