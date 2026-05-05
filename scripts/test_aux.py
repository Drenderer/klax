import equinox as eqx
import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random as jr

import klax

key = jr.key(0)
model_key, train_key, aux_key = jr.split(key, 3)

model = klax.nn.MLP("scalar", "scalar", [16], key=model_key)


class UpdateAux(klax.Callback):
    @eqx.filter_jit
    def on_training_step(self, context):
        context.state.run_state, _ = jr.split(context.state.run_state)


@klax.loss
def my_loss(model, batch, run_state):
    key = run_state
    x = jr.normal(key, (64,))
    y = jnp.sin(x)
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y - y_pred) ** 2)


model, history = klax.fit(
    model,
    data=None,
    steps=30_000,
    callbacks=[UpdateAux()],
    run_state=aux_key,
    loss=my_loss,
    key=train_key,
)

x = jnp.linspace(-4, 4, 100)
y = jnp.sin(x)

ax = history.plot("loss")
plt.show()
y_pred = jax.vmap(model)(x)
fig, ax = plt.subplots()
ax.plot(x, y, label="True", marker="o", markevery=10)
ax.plot(x, y_pred, label="Predicted")
ax.legend()
plt.show()
