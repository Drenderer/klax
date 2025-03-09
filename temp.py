import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import klax

x = jnp.linspace(0, 1.0, 10)[:, jnp.newaxis]
y = 2.*x + 1.0
model = eqx.nn.Linear(1, 1, key=jr.key(0))
model, history = klax.fit(model, x, y, steps=10_000, key=jr.key(0))
print(model.weight)
print(model.bias)