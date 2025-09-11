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

# # Add two zoom-in spy lenses (lower left and lower right)

# # Lower left zoom
# axins1 = inset_axes(ax, width="30%", height="30%", loc="lower left", borderpad=2)
# for name, result in results.items():
#     history = result["history"]
#     axins1.plot(history.steps, history.loss, label=f"{name} loss")
#     axins1.plot(history.steps, history.val_loss, ls="--")
# # Set zoomed region (adjust as needed)
# axins1.set_xlim(history.steps[0], history.steps[len(history.steps)//10])
# axins1.set_ylim(min(history.loss[:len(history.steps)//10]), max(history.loss[:len(history.steps)//10]))
# axins1.set_yscale("log")
# axins1.set_xticks([])
# axins1.set_yticks([])

# # Lower right zoom
# axins2 = inset_axes(ax, width="30%", height="30%", loc="lower right", borderpad=2)
# for name, result in results.items():
#     history = result["history"]
#     axins2.plot(history.steps, history.loss, label=f"{name} loss")
#     axins2.plot(history.steps, history.val_loss, ls="--")
# # Set zoomed region (adjust as needed)
# axins2.set_xlim(history.steps[-len(history.steps)//10], history.steps[-1])
# axins2.set_ylim(min(history.loss[-len(history.steps)//10:]), max(history.loss[-len(history.steps)//10:]))
# axins2.set_yscale("log")
# axins2.set_xticks([])
# axins2.set_yticks([])

# # Draw lines between main plot and insets
# mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.5")
# mark_inset(ax, axins2, loc1=1, loc2=3, fc="none", ec="0.5")
