# %% Imports

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import numpy as jnp
from jax import random as jr
from mpl_toolkits.mplot3d import Axes3D

import klax


# %% Generate some dummy data
def keygen(key):
    while True:
        key, new_key = jr.split(key)
        yield new_key


key = keygen(jr.key(0))

n = 5
A = jr.normal(next(key), (n, n))
A = A @ A.T  # Ensure positive semi-definiteness
B = jr.normal(next(key), (n, n))
B = -B @ B.T


def func(x, p):
    y = jnp.einsum("...k,...k->...", x, jnp.matmul(x, A))
    u = jnp.einsum("...k,...k->...", p, jnp.matmul(p, B))
    return y + u


xs = -1 + 2 * jr.uniform(next(key), (10000, n))
ps = -1 + 2 * jr.uniform(next(key), (10000, n))
ys = func(xs, ps)

# %% Try to plot the function if the dimensionality allows

if n == 1:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Create a meshgrid for surface plotting
    X, P = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    Y = func(
        X[..., None], P[..., None]
    )  # Calculate Y values using the function

    # Create the surface plot
    surf = ax.plot_surface(X, P, Y, cmap="viridis", edgecolor="none")

    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_zlabel("y")
    ax.set_title("Function Surface Plot")

    plt.colorbar(surf, ax=ax, label="y value")
    plt.show()

# %% Fit the PICNN

picnn = klax.nn.PICNN(
    x_size=n,
    p_size=n,
    out_size="scalar",
    width_sizes=[64, 64],
    key=next(key),
)


class BaselineMLP(eqx.Module):
    mlp: klax.nn.MLP

    def __init__(self, n):
        self.mlp = klax.nn.MLP(
            in_size=2 * n,
            out_size="scalar",
            width_sizes=[64, 64],
            key=next(key),
        )

    def __call__(self, x, p):
        y = jnp.concat([x, p])
        return self.mlp(y)


mlp = BaselineMLP(n)

# %% Try to plot the model if the dimensionality allows

if n == 1:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    X, P = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    Y = jax.vmap(jax.vmap(klax.finalize(picnn)))(X[..., None], P[..., None])

    surf = ax.plot_surface(X, P, Y, cmap="viridis", edgecolor="none")

    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.set_zlabel("y")
    ax.set_title("Function Surface Plot")

    plt.colorbar(surf, ax=ax, label="y value")
    plt.show()

# %% Fit the PICNN


@klax.loss
def loss(model, batch, batch_axes):
    xs, ps, ys = batch
    ys_pred = jax.vmap(model)(xs, ps)
    return jnp.mean(jnp.square(ys - ys_pred))


picnn, hist = klax.fit(
    picnn,
    (xs, ps, ys),
    loss=loss,
    optimizer=optax.adam(5e-4),
    steps=50_000,
    key=next(key),
)

hist.plot()


mlp, hist = klax.fit(
    mlp,
    (xs, ps, ys),
    loss=loss,
    optimizer=optax.adam(5e-4),
    steps=50_000,
    key=next(key),
)

hist.plot()
