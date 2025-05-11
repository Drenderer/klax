"""
Testing if the non-negative wrapper can recover from a negative initial value.
I.e., if the gradients vanish when the weights are negative or zero.
"""

# %% Imports
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from klax import fit
from klax import NonNegative, unwrap


# %% Define a simple model
class SimpleModel(eqx.Module):
    weight: jax.Array

    def __init__(self):
        self.weight = NonNegative(jnp.array(-1.0))

    def __call__(self, x):
        return self.weight * x


# %% Generate data
def fun(x):
    return -2 * x


x = jnp.linspace(-1, 1, 100)
y = jax.vmap(fun)(x)

# %% Train the model
model = SimpleModel()

# Let the optimizer run the parameter into the negatives
print("Initial weight:", unwrap(model).weight)

model, hist = fit(model, (x, y), steps=10000, key=jr.key(0))


# Redefine the data
def fun(x):
    return 2 * x


y = jax.vmap(fun)(x)

# Train the model to test recovery
model, hist = fit(model, (x, y), steps=10000, key=jr.key(0))

print("Final weight:", unwrap(model).weight)
