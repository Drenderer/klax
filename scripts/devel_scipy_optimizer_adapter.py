# %% Imports
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random as jr
from jaxtyping import Array, PyTree
from scipy.optimize import minimize

import klax

jax.config.update("jax_enable_x64", True)

# %% How to flatten a klax model

key = jr.key(0)
model = klax.nn.MLP("scalar", "scalar", [8, 8], key=key)


# bounds = jax.tree.map(lambda x:)
def model_to_flat_array(model: PyTree) -> tuple[Array, list, PyTree]:
    model_params, model_static = eqx.partition(model, eqx.is_inexact_array)
    model_params_leaves, model_params_treedef = jax.tree.flatten(model_params)

    model_params_shapes = jax.tree.map(
        lambda x: x.shape if isinstance(x, Array) else None,
        model_params_leaves,
    )
    model_params_values = jnp.concat(
        jax.tree.map(lambda x: jnp.reshape(x, -1), model_params_leaves)
    )
    return (
        model_params_values,
        model_params_shapes,
        model_params_treedef,
        model_static,
    )


def flat_array_to_model(
    model_params_values,
    model_params_shapes,
    model_params_treedef,
    model_static,
):
    model_params_sizes = jax.tree.map(
        lambda x: jnp.prod(jnp.array(x)),
        model_params_shapes,
        is_leaf=lambda x: isinstance(x, tuple),
    )

    model_params_values = jnp.split(
        model_params_values, jnp.cumsum(jnp.stack(model_params_sizes))
    )[:-1]
    model_params_leaves = jax.tree.map(
        lambda x, s: x.reshape(s), model_params_values, model_params_shapes
    )
    model_params = jax.tree.unflatten(
        model_params_treedef, model_params_leaves
    )
    return eqx.combine(model_params, model_static)


values, shapes, treedef, static = model_to_flat_array(model)
_model = flat_array_to_model(values, shapes, treedef, static)
assert eqx.tree_equal(_model, model)


def scipy_loss_wrapper(loss: klax.loss, model, data):
    values, shapes, treedef, static = model_to_flat_array(model)

    def wrapped_loss(values):
        values = jnp.asarray(values)
        model = flat_array_to_model(values, shapes, treedef, static)
        loss_value, grad = loss.value_and_grad(model, data, None)
        grad, _, _, _ = model_to_flat_array(grad)
        return np.array(loss_value), np.array(grad)

    return wrapped_loss


# %%

x = jnp.linspace(-6, 6, 100)
y = jnp.sin(x) + 0.2 * jr.normal(key, x.shape)
data = (x, y)

scipy_loss = scipy_loss_wrapper(klax.mse, model, data)
values, shapes, treedef, static = model_to_flat_array(model)

sopt_result = minimize(
    fun=scipy_loss,
    x0=values,
    jac=True,
    tol=0,
    method="L-BFGS-B",
    options={"maxiter": 1000, "ftol": 1e-12, "disp": True},
    bounds=None,
    constraints=(),
)
optimized_values = sopt_result.x

# %%
model = flat_array_to_model(optimized_values, shapes, treedef, static)
y_pred = jax.vmap(model)(x)

plt.scatter(x, y)
plt.plot(x, y_pred)
