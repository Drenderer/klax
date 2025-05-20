# Copyright 2025 The Klax Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Testing if the non-negative wrapper can recover from a negative initial value.
I.e., if the gradients vanish when the weights are negative or zero.
"""

# %% Imports
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from paramax import unwrap

from klax import fit, NonNegative, apply



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
print("Initial parameter:", model.weight.parameter)

model = apply(model)

print("Parameter after applying wrapper:", model.weight.parameter)

model, hist = fit(
    model,
    (x, y),
    steps=10000,
    key = jr.key(0)
)

print("Final weight:", unwrap(model).weight)
print("Final parameter:", model.weight.parameter)


