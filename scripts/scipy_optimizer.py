# %% Imports
import time
from collections.abc import Sequence
from contextlib import contextmanager

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


class ScipyLogger(klax.Callback):
    history: klax.History
    metrics: dict[str, klax.Metric]
    start_time: float = 0.0

    def __init__(
        self,
        metrics: Sequence[klax.Metric] | None = None,
    ):
        self.metrics = {} if metrics is None else {m.name: m for m in metrics}
        self.history = klax.History()

    def on_training_step(self, context: klax.TrainingContext) -> None:
        for metric in self.metrics.values():
            metric_value = jax.device_get(metric(context))
            self.history.append(context.state.step, metric.name, metric_value)

    def on_training_start(self, context: klax.TrainingContext) -> None:
        self.start_time = time.time()
        self.on_training_step(context)

    def on_training_end(self, context: klax.TrainingContext) -> None:
        end_time = time.time()
        self.history.total_time = end_time - self.start_time
        self.history.total_steps = context.state.step


logger = ScipyLogger(
    metrics=[
        klax.BatchMetric(
            "loss",
            klax.mse,
            data,
            batcher=klax.batch_data,
            batch_size=1000,
            key=jr.key(0),
        )
    ]
)

# %% Optimize using scipy
model = klax.nn.FICNN("scalar", "scalar", [8, 8], key=key)
with timer("SLSQP training"):
    slsqp_model, _ = klax.scipy_fit(
        model, data, loss=klax.mse, callbacks=[logger], verbose=True
    )

# %% Optimize using Adam
adam_model = klax.nn.FICNN("scalar", "scalar", [8, 8], key=key)
with timer("Adam training"):
    adam_model, hist = klax.fit(
        model, (x, y), loss=klax.mse, steps=10_000, key=key
    )

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
