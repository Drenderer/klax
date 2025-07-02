# Copyright 2025 The Klax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""No Data Training Example.

This example demonstrates how to train a model without any data.
This is useful for training PINNs or other models that do not require data.
"""

# %% Imports
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import grad, vmap
from jaxtyping import Array, PRNGKeyArray
from matplotlib import pyplot as plt

import klax

# %% Define a PINN-style model


class PINN(eqx.Module):
    """A simple PINN-style model that solves the ODE u_x + u = 0 with u(0) = 1."""

    mlp: klax.nn.MLP
    xs: Array  # evalutaion points

    def __init__(self, xs: Array, key: PRNGKeyArray):
        self.xs = xs
        self.mlp = klax.nn.MLP(
            "scalar", "scalar", 2 * [16], activation=jax.nn.softplus, key=key
        )

    def __call__(self, x):
        return self.mlp(x)

    @staticmethod
    def residural_loss(model, batch, batch_axis):
        """Residual loss definition.
        
        We define a loss function that penalizes the
        residual of the ODE u_x + u = 0 and violations
        of the boundary conditions u(0) = 1.
        """
        xs = jax.lax.stop_gradient(model.xs)

        # ODE residual
        u = vmap(model)(xs)
        u_xx = vmap(grad(model))(xs)
        residual = u_xx + u
        residual_loss = jnp.mean(residual**2)

        # Boundary conditions
        bc_0 = model(jnp.array(0.0)) - 1.0
        bc_loss = bc_0**2

        return residual_loss + 0.1 * bc_loss


# %% Train the model without data

xs = jnp.linspace(0, 1, 100)
model = PINN(xs, jr.key(0))

model, history = klax.fit(
    model,
    data=None,
    batch_axis=None,
    steps=100_000,
    optimizer=optax.adam(1e-5),
    loss_fn=model.residural_loss,
    key=jr.key(1),
)

history.plot()

# %% Plot the result


def solution(x):
    """Return the true solution of the ODE u_x + u = 0 with u(0) = 1."""
    return jnp.exp(-x)


x = jnp.linspace(0, 4, 1000)
u_true = vmap(solution)(x)
u = vmap(model)(x)
u_x = vmap(grad(model))(x)
residual = u_x + u

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(x, u_true, marker="o", markevery=100, label="u_true(x)")
axes[0].plot(x, u, label="PINN")
axes[0].set(
    title="PINN solution",
    xlabel="x",
    ylabel="u(x)",
)
axes[0].axvspan(model.xs.min(), model.xs.max(), color='gray', alpha=0.2, label="Training region")
axes[0].legend()

axes[1].plot(x, residual, label="residual")
axes[1].set(
    title="Residual of ODE",
    xlabel="x",
    ylabel="residual",
)
axes[1].axvspan(model.xs.min(), model.xs.max(), color='gray', alpha=0.2, label="Training region")
axes[1].legend()

plt.tight_layout()
plt.show()
