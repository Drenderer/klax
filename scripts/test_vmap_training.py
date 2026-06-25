# %% Imports
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from tqdm.auto import tqdm

import klax

# %% Define a model

key = jr.key(0)


@eqx.filter_vmap
def make_ensemble(key):
    return klax.nn.MLP("scalar", "scalar", [16, 16], key=key)


models = make_ensemble(jr.split(key, 10))

# %% Define some data

x = jnp.linspace(0, 3, 1000)
y = jnp.sin(x)

data = (x, y)

# %% Define a custom training loop

loss = klax.mse


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(eqx.if_array(0), None, None, None))
def make_step(state_leaves, state_treedef, batch, static):
    state = jax.tree.unflatten(state_treedef, state_leaves)
    model, opt_state = state

    optimizer, loss = static

    value, grad = loss.value_and_grad(model, batch, None)

    model_params, model_static = eqx.partition(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grad, opt_state, model_params)

    model_params = optax.apply_updates(model_params, updates)
    model = eqx.combine(model_params, model_static)

    state = model, opt_state
    state_leaves, state_treedef = jax.tree.flatten(state)

    return state_leaves


optimizer = optax.adam(1e-3)
opt_states = jax.vmap(optimizer.init)(eqx.filter(models, eqx.is_inexact_array))
states = (models, opt_states)
static = (optimizer, loss)

states_leaves, states_treedef = jax.tree.flatten(states)
for step in tqdm(range(1000)):
    states_leaves = make_step(states_leaves, states_treedef, data, static)
states = jax.tree.unflatten(states_treedef, states_leaves)

models, opt_state = states

# %% Evaluate ensemble


@eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
def evaluate_ensemble(model, x):
    return model(x)


x_test = jnp.linspace(-3, 6, 1000)
ys_pred = jax.vmap(evaluate_ensemble, in_axes=(None, 0))(models, x_test)

plt.axvspan(0, 3, color="lightgray", alpha=0.3)
plt.plot(x_test, ys_pred)
plt.plot(x_test, jnp.sin(x_test), color="black", marker="o", markevery=100)
