# %% Imports
import time
from collections.abc import Sequence
from contextlib import contextmanager

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random as jr

import klax

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

# %% Define a callback

dataset_loss = klax.BatchMetric(
    "loss",
    klax.mse,
    data,
    batcher=klax.batch_data,
    batch_size=100,
    key=jr.key(0),
)

logger = klax.MetricLogger(
    log_every=10,
    metrics=[dataset_loss],
)

# %% Optimize using scipy
model = klax.nn.FICNN("scalar", "scalar", [8, 8], key=key)
with timer("SLSQP training"):
    slsqp_model, _ = klax.scipy_fit(
        model, data, loss=klax.mse, callbacks=[logger], verbose=True
    )

logger.history.plot()
plt.show()
# %% Optimize using Adam
adam_model = klax.nn.FICNN("scalar", "scalar", [8, 8], key=key)
with timer("Adam training"):
    adam_model, hist = klax.fit(
        model, (x, y), loss=klax.mse, steps=10_000, batch_size=100, key=key
    )

hist.plot()
plt.show()
# %% Plot result
slsqp_model = klax.finalize(slsqp_model)
slsqp_y_pred = jax.vmap(slsqp_model)(x)

adam_model = klax.finalize(adam_model)
adam_y_pred = jax.vmap(adam_model)(x)

fig, ax = plt.subplots()
ax.scatter(x, y, c="grey", marker="x", label="Data")
ax.plot(x, slsqp_y_pred, label="SLSQP")
ax.plot(x, adam_y_pred, label="ADAM")
ax.set(xlabel="x", ylabel="y", title="FICNN optimizer comparison")
ax.legend()
plt.show()
