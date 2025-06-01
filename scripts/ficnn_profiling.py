"""
Testing the computational performance of FICNN and comparing it to alternative implementations.
"""

# %% Imports

from __future__ import annotations
from collections.abc import Callable, Sequence
from typing import (
    Literal,
    Self,
)

import timeit

from matplotlib import pyplot as plt

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, zeros, he_normal
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

import klax
from klax._misc import default_floating_dtype
from klax.nn import FICNN, MLP, Linear
from klax import fit
from klax import HistoryCallback
from klax import Constraint, NonNegative


# %% Custom code for timeing code
def format_time(seconds: float) -> str:
    units = [
        ("s", 1),
        ("ms", 1e-3),
        ("µs", 1e-6),
        ("ns", 1e-9),
    ]

    # Sort from largest to smallest unit
    for unit_name, unit_value in units:
        if seconds >= unit_value:
            value = seconds / unit_value
            return f"{value:.3f} {unit_name}"

    # If it's even smaller than ns
    return f"{seconds / 1e-12:.0f} ps"


def time_code(func, msg=""):
    n, t1 = timeit.Timer(func).autorange()
    repeat = 6 if t1 < 1 else 2
    timer = timeit.repeat(func, repeat=repeat, number=n)
    timer += [t1]
    timer = jnp.array(timer) / n
    mean = format_time(float(jnp.mean(timer)))
    std = format_time(float(jnp.std(timer)))
    print(
        msg
        + f"{mean} ± {std} per loop (mean ± std. dev. of {timer.size} runs, {n} loops each)"
    )


# %% Define alternative FICNN implementations


class PartialInputNonNegative(Constraint):
    parameter: Array
    n: int

    def __init__(self, parameter: Array, n: int):
        # Ensure that the parameter fulfills the constraint initially
        self.parameter = self._apply_constraint(klax.unwrap(parameter), n)
        self.n = n

    @staticmethod
    def _apply_constraint(x: Array, n: int) -> Array:
        # Apply non-negative constraint to the first n rows
        constrained = jnp.where(
            jnp.arange(x.shape[0])[:, None] < n, jnp.maximum(x, 0), x
        )
        return constrained

    def unwrap(self) -> Array:
        return self.parameter

    def apply(self) -> Self:
        return eqx.tree_at(
            lambda t: t.parameter,
            self,
            self._apply_constraint(self.parameter, self.n),
        )


class HalfConstrainedFICNN(eqx.Module):
    layers: tuple[Linear, ...]
    activations: tuple[Callable, ...]
    final_activation: Callable
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    use_passthrough: bool = eqx.field(static=True)
    non_decreasing: bool = eqx.field(static=True)
    in_size: int | Literal["scalar"] = eqx.field(static=True)
    out_size: int | Literal["scalar"] = eqx.field(static=True)
    width_sizes: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        out_size: int | Literal["scalar"],
        width_sizes: Sequence[int],
        use_passthrough: bool = True,
        non_decreasing: bool = False,
        weight_init: Initializer = he_normal(),
        bias_init: Initializer = zeros,
        activation: Callable = jax.nn.softplus,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype: type | None = None,
        *,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        width_sizes = tuple(width_sizes)

        self.in_size = in_size
        self.out_size = out_size
        self.width_sizes = width_sizes
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias
        self.use_passthrough = use_passthrough
        self.non_decreasing = non_decreasing

        in_size = 1 if in_size == "scalar" else in_size
        in_sizes = (in_size,) + width_sizes
        out_sizes = width_sizes + (out_size,)
        use_biases = len(width_sizes) * (use_bias,) + (use_final_bias,)
        keys = jr.split(key, len(in_sizes))

        layers = []
        for n, (sin, sout, ub, key) in enumerate(
            zip(in_sizes, out_sizes, use_biases, keys)
        ):
            if n == 0:
                _sin = sin
                constr = NonNegative if non_decreasing else None
            else:
                _sin = sin + in_size if use_passthrough else sin
                if non_decreasing:
                    constr = NonNegative
                else:
                    # Define a constraint type that captures n=sin
                    class _PartialInputNonNegative(PartialInputNonNegative):
                        def __new__(cls, parameter):
                            return PartialInputNonNegative(parameter, n=sin)

                    constr = _PartialInputNonNegative

            layers.append(
                Linear(
                    _sin,
                    sout,
                    weight_init,
                    bias_init,
                    ub,
                    constr,
                    dtype=dtype,
                    key=key,
                )
            )

        self.layers = tuple(layers)

        # In case `activation` or `final_activation` are learnt, then make a separate
        # copy of their weights for every neuron.
        activations = []
        for width in width_sizes:
            activations.append(
                eqx.filter_vmap(lambda: activation, axis_size=width)()
            )
        self.activations = tuple(activations)
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = eqx.filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()

    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
        """
        Args:
            x: A JAX array with shape `(in_size,)`. (Or shape `()` if
                `in_size="scalar"`.)
            key: Ignored; provided for compatibility with the rest of the
                Equinox API. (Keyword only argument.)

        Returns:
            A JAX array with shape `(out_size,)`. (Or shape `()` if
            `out_size="scalar"`.)
        """

        y = self.layers[0](x)

        for i, (layer, activation) in enumerate(
            zip(self.layers[1:], self.activations)
        ):
            layer_activation = jax.tree.map(
                lambda y: y[i] if eqx.is_array(y) else y, activation
            )
            y = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, y)

            if self.use_passthrough:
                y = jnp.concat([y, x], axis=-1)
            y = layer(y)

        if self.out_size == "scalar":
            y = self.final_activation(y)
        else:
            y = eqx.filter_vmap(lambda a, b: a(b))(self.final_activation, y)

        return y


# %% Define some data
key = jr.key(0)

in_size = 3

x = jr.normal(key, (1000, in_size))
y = jax.vmap(jnp.inner)(x, x) + 0.1 * jr.normal(key, shape=(x.shape[0],))

# %% Test the alternateive implementation
for use_passthrough, non_decreasing in [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
]:
    ficnn = klax.finalize(
        HalfConstrainedFICNN(
            in_size,
            "scalar",
            1 * [8],
            use_passthrough=use_passthrough,
            non_decreasing=non_decreasing,
            key=key,
        )
    )
    assert ficnn(x[0]).shape == (), "Unexpected output shape"
    if non_decreasing:
        ficnn_x = jax.vmap(jax.grad(ficnn))
        assert jnp.all(ficnn_x(x) >= 0), (
            "FICNN(..., non_decreasing=True) is not non-decreasing."
        )
    ficnn_xx = jax.vmap(jax.hessian(ficnn))
    assert jnp.all(jnp.linalg.eigvals(ficnn_xx(x)) >= 0), (
        "FICNN(...) is not convex."
    )

# %% Define models

kwargs = dict(
    in_size=in_size,
    out_size="scalar",
    width_sizes=1 * [4],
    use_passthrough=True,
    non_decreasing=False,
    key=key,
)

hc_ficnn = HalfConstrainedFICNN(**kwargs)  # pyright: ignore
ficnn = FICNN(**kwargs)  # pyright: ignore


# %% Time the un-vmapped call time
print("\nTiming un-vmapped call time with finalize")
time_code(lambda: klax.finalize(hc_ficnn)(x[0]), "HCFICNN: ")
time_code(lambda: klax.finalize(ficnn)(x[0]), "FICNN:   ")

_hc_ficnn = klax.finalize(hc_ficnn)
_ficnn = klax.finalize(ficnn)
print("\nTiming un-vmapped call time without finalize")
time_code(lambda: _hc_ficnn(x[0]), "HCFICNN: ")
time_code(lambda: _ficnn(x[0]), "FICNN:   ")

# %% Time the vmapped call time
_hc_ficnn = jax.vmap(klax.finalize(hc_ficnn))
_hc_ficnn(x[:1])  # Call to compile
_ficnn = jax.vmap(klax.finalize(ficnn))
_ficnn(x[:1])  # Call to compile
print("\nTiming vmapped call time")
time_code(lambda: _hc_ficnn(x), "HCFICNN: ")
time_code(lambda: _ficnn(x), "FICNN:   ")

# %% Time training time
print("\nTiming training time")
time_code(
    lambda: fit(
        hc_ficnn,
        (x, y),
        steps=10_000,
        history=HistoryCallback(verbose=False),
        key=key,
    ),
    "HCFICNN: ",
)
time_code(
    lambda: fit(
        ficnn,
        (x, y),
        steps=10_000,
        history=HistoryCallback(verbose=False),
        key=key,
    ),
    "FICNN:   ",
)


# %% Sanity check
x = jnp.linspace(-2, 2, 20)[:, None]
x_eval = jnp.linspace(-2, 3, 1000)[:, None]
y = jax.vmap(jnp.inner)(x, x) + 0.3 * jr.normal(key, shape=(x.shape[0],))

kwargs = dict(
    in_size=1,
    out_size="scalar",
    width_sizes=2 * [8],
    use_passthrough=True,
    non_decreasing=False,
    key=key,
)
hc_ficnn = HalfConstrainedFICNN(**kwargs)  # pyright: ignore
ficnn = FICNN(**kwargs)  # pyright: ignore
mlp = MLP(in_size=1, out_size="scalar", width_sizes=2 * [8], key=key)
eqx_mlp = eqx.nn.MLP(
    in_size=1,
    out_size="scalar",
    width_size=8,
    depth=2,
    activation=jax.nn.softplus,
    key=key,
)

_hc_ficnn, hist = fit(
    hc_ficnn,
    (x, y),
    steps=20_000,
    history=HistoryCallback(verbose=False),
    key=key,
)
_ficnn, hist = fit(
    ficnn,
    (x, y),
    steps=20_000,
    history=HistoryCallback(verbose=False),
    key=key,
)
_mlp, hist = fit(
    mlp, (x, y), steps=20_000, history=HistoryCallback(verbose=False), key=key
)
_eqx_mlp, hist = fit(
    eqx_mlp,
    (x, y),
    steps=20_000,
    history=HistoryCallback(verbose=False),
    key=key,
)

fig, ax = plt.subplots()
ax.scatter(x, y, label="Data", marker="x", c="black")
ax.plot(
    x_eval, jax.vmap(klax.finalize(_eqx_mlp))(x_eval), ls="-.", label="EQX MLP"
)
ax.plot(x_eval, jax.vmap(klax.finalize(_mlp))(x_eval), ls="-.", label="MLP")
ax.plot(x_eval, jax.vmap(klax.finalize(_hc_ficnn))(x_eval), label="HCFICNN")
ax.plot(
    x_eval, jax.vmap(klax.finalize(_ficnn))(x_eval), ls="--", label="FICNN"
)
ax.legend()
plt.show()
