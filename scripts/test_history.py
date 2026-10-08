# %%
from typing import Any

from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from klax import History

# %%
random_array = jr.uniform(key=jr.key(0), shape=(3, 2, 10))


# %%
def plot(self, *keys: str, ax: Any = None, **kwargs: Any) -> None:
    # plt = get_plot()

    if ax is None:
        _, ax = plt.subplots()
        ax.set(
            xlabel="Step",
            ylabel="Metric",
            yscale="log",
            title="Training History",
        )
        ax.grid(True)
    keys = keys if keys else self.keys()
    artists = []
    for name in keys:
        steps, values = self.content[name]
        values = jnp.stack(values, axis=0)
        if values.ndim > 2:
            values = values.reshape(values.shape[0], -1)
        artist = ax.plot(steps, values, **kwargs)
        artist = artist[0] if len(artist) == 1 else tuple(artist)
        artists.append(artist)
    ax.legend(
        artists,
        keys,
        handler_map={tuple: HandlerTuple(ndivide=6, pad=0)},
    )
    return ax


# %%
history = History()
history.append("mse", 1, 0.2)
history.append("mse", 2, 0.1)
history.append("mse", 3, 0.3)
history.append("array", 1, random_array[0])
history.append("array", 2, random_array[1])
history.append("array", 3, random_array[2])


plot(history, "array", marker="o")
