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
from klax.wrappers import NonNegative, unwrap


# %% Define a simple model
class SimpleModel(eqx.Module):
    weight: jax.Array

    def __init__(self):
        self.weight = NonNegative(jnp.array(-1.))

    def __call__(self, x):
        return self.weight * x
    
# %% Generate data
def fun(x):
    return 2*x

x = jnp.linspace(-1, 1, 100)
y = jax.vmap(fun)(x)

# %% Train the model
model = SimpleModel()

# Do model surgery to make the wrapped array negative
# model = eqx.tree_at(
#     lambda m: m.weight.parameter, model, jnp.array(-1e-10)
# )

print("Initial weight:", unwrap(model).weight)

model, hist = fit(
    model,
    (x, y),
    steps=10000,
    key = jr.key(0)
)

print("Final weight:", unwrap(model).weight)


