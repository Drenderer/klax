# %% Imports
import itertools
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random as jr
from jaxtyping import Array, PyTree, PyTreeDef
from scipy.optimize import OptimizeResult, minimize

import klax

# jax.config.update("jax_enable_x64", True)

# %% Define some evaluation utils


@contextmanager
def timer(name=""):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.2f}s")


# %% Define data

key = jr.key(0)
x = jnp.linspace(-4, 2, 100)
y = jnp.sin(x) + 0.01 * jr.normal(key, x.shape)
data = (x, y)

# %% Optimize using scipy
model = klax.nn.FICNN("scalar", "scalar", [8, 8], key=key)
with timer("SLSQP training"):
    slsqp_model, _ = klax.scipy_fit(model, data, verbose=True)

# %% Optimize using Adam
adam_model = klax.nn.FICNN("scalar", "scalar", [8, 8], key=key)
with timer("Adam training"):
    adam_model, hist = klax.fit(model, (x, y), steps=10_000, key=key)

# %% Plot result
slsqp_model = klax.finalize(slsqp_model)
slsqp_y_pred = jax.vmap(slsqp_model)(x)

adam_model = klax.finalize(adam_model)
adam_y_pred = jax.vmap(adam_model)(x)

fig, ax = plt.subplots()
ax.scatter(x, y, c="grey")
ax.plot(x, slsqp_y_pred, label="SLSQP")
ax.plot(x, adam_y_pred, label="ADAM")
ax.set(xlabel="x", ylabel="y", title="Optimizer comparison")
ax.legend()
plt.show()
