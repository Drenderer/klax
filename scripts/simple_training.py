import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random as jr

import klax

key = jr.key(0)
model_key, train_key = jr.split(key, 2)

x = jnp.linspace(-3, 3, 1000)
y = jnp.sin(x)

model = klax.nn.MLP("scalar", "scalar", [16], key=model_key)

model, history = klax.fit(
    model,
    (x, y),
    validation_data=(x, y),
    steps=30_000,
    key=train_key,
)


plt.plot(x, y, label="True", lw=3, ls="--")
plt.plot(x, jax.vmap(model)(x), color="red", label="Predicted")
plt.legend()
plt.title("Sine Function Approximation with klax")
plt.show()
