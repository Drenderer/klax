"""Compare the initialization from Hoedt and Klambauer to standard initialization schemes."""

# %% Imports

import jax
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

import klax

key = jr.key(0)
data_key, model_key, train_key = jr.split(key, 3)

# %% Generate some dummy data

in_size = 10
noise_amplitude = 0.5

a_key, x_key, y_key = jr.split(data_key, 3)
A = klax.nn.ConstantSPDMatrix(in_size, key=a_key)(jnp.array(()))

x = jr.uniform(x_key, shape=(1000, in_size))
y = jax.vmap(lambda x: jnp.inner(x, A @ x))(x)
y += noise_amplitude * jr.normal(y_key, shape=y.shape)

# %% Fit two FICNNs

configs = (
    ("baseline", False, None, None),
    ("Hoedt_init", False, klax.hoedt_normal(), klax.hoedt_bias()),
    ("baseline_passthrough", True, None, None),
    ("Hoedt_init_passthrough", True, klax.hoedt_normal(), klax.hoedt_bias()),
)

results = dict()
for (
    name,
    use_passthrough,
    constrained_weight_init,
    constrained_bias_init,
) in configs:
    ficnn = klax.nn.FICNN(
        in_size,
        "scalar",
        width_sizes=[64, 64],
        use_passthrough=use_passthrough,
        constrained_weight_init=constrained_weight_init,
        constrained_bias_init=constrained_bias_init,
        key=model_key,
    )

    ficnn, history = klax.fit(
        ficnn,
        (x, y),
        steps=100_000,
        history=klax.HistoryCallback(log_every=10),
        key=train_key,
    )
    results[name] = dict(ficnn=ficnn, history=history)

# %% Evaluate

fig, ax = plt.subplots()

for name, result in results.items():
    history = result["history"]
    (line,) = ax.plot(history.steps, history.loss, label=f"{name} loss")
    ax.plot(
        history.steps,
        history.val_loss,
        label=f"{name} val loss",
        c=line.get_color(),
        ls="--",
    )

ax.set(
    yscale="log",
    xlabel="Steps",
    ylabel="Loss",
    title="FICNN initialization comparison",
    xlim=[0, 10000],
)
ax.legend()
