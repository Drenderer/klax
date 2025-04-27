"""
Comparison of SPDMatrix to an alternative implementation
"""

# %% Imports
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn.initializers import Initializer, he_normal, zeros

from typing import Literal, Sequence, Any, Callable, Optional
from jaxtyping import PRNGKeyArray, Array

import timeit

from klax.nn import MLP


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
    mean = format_time(jnp.mean(timer))
    std = format_time(jnp.std(timer))
    print(
        msg
        + f"{mean} ± {std} per loop (mean ± std. dev. of {timer.size} runs, {n} loops each)"
    )

# %% Define implementations

class OriginalSPDMatrix(eqx.Module):

    func: MLP
    shape: tuple[int] = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    _tensor: Array  # Not learnable transform tensor from the vector-space of components to the space of lower triangular matrices

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        shape: Optional[int | tuple[int]] = None,
        epsilon: float = 1e-6,
        width_sizes: Optional[Sequence[int]] = None,
        weight_init: Initializer = he_normal(),
        bias_init: Initializer = zeros,
        activation: Callable = jax.nn.softplus,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype: Any | None = None,
        *,
        key: PRNGKeyArray,
    ):
        in_size_ = 1 if in_size == "scalar" else in_size
        shape = in_size_ if shape is None else shape
        width_sizes = (
            [
                int(2**(jnp.ceil(jnp.log2(in_size_)))),
            ]
            if width_sizes is None
            else width_sizes
        )
        shape = shape if isinstance(shape, tuple) else 2 * (shape,)
        if shape[-1] != shape[-2]:
            raise ValueError(
                "The last two dimensions in shape must be equal for symmetric matrices."
            )

        # Construct the tensor
        n = shape[-1]
        num_batches = int(jnp.prod(jnp.array(shape[:-2])))
        num_elements = n * (n + 1) // 2
        tensor = jnp.zeros((n, n, num_elements), dtype="int32")
        for e, (i, j) in enumerate(zip(*jnp.tril_indices(n))):
            tensor = tensor.at[i, j, e].set(1)

        self._tensor = tensor
        self.shape = shape
        self.epsilon = epsilon
        self.func = MLP(
            in_size,
            num_elements * num_batches,
            width_sizes,
            weight_init,
            bias_init,
            activation,
            final_activation,
            use_bias,
            use_final_bias,
            dtype,
            key=key,
        )

    def __call__(self, x: Array) -> Array:
        elements = self.func(x).reshape(*self.shape[:-2], -1)
        L = jnp.einsum("ijk,...k->...ij", jax.lax.stop_gradient(self._tensor), elements)
        A = L @ jnp.conjugate(L.mT)
        identity = jnp.broadcast_to(jnp.eye(self.shape[-1]), A.shape)
        return A + self.epsilon * identity


class AlternativeSPDMatrix(eqx.Module):

    func: MLP
    shape: tuple[int] = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        shape: Optional[int | tuple[int]] = None,
        epsilon: float = 1e-6,
        width_sizes: Optional[Sequence[int]] = None,
        weight_init: Initializer = he_normal(),
        bias_init: Initializer = zeros,
        activation: Callable = jax.nn.softplus,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype: Any | None = None,
        *,
        key: PRNGKeyArray,
    ):
        in_size_ = 1 if in_size == "scalar" else in_size
        shape = in_size_ if shape is None else shape
        width_sizes = (
            [
                int(2**(jnp.ceil(jnp.log2(in_size_)))),
            ]
            if width_sizes is None
            else width_sizes
        )
        shape = shape if isinstance(shape, tuple) else 2 * (shape,)
        if shape[-1] != shape[-2]:
            raise ValueError(
                "The last two dimensions in shape must be equal for symmetric matrices."
            )

        num_batches = int(jnp.prod(jnp.array(shape)))

        self.shape = shape
        self.epsilon = epsilon
        self.func = MLP(
            in_size,
            num_batches,
            width_sizes,
            weight_init,
            bias_init,
            activation,
            final_activation,
            use_bias,
            use_final_bias,
            dtype,
            key=key,
        )

    def __call__(self, x: Array) -> Array:
        L = self.func(x).reshape(self.shape)
        A = L @ jnp.conjugate(L.mT)
        identity = jnp.broadcast_to(jnp.eye(self.shape[-1]), A.shape)
        return A + self.epsilon * identity
    

# %% Time the different versions

key = jr.key(0)
model_key, data_key = jr.split(key)

n = 50
N = 50

x = jr.normal(data_key, (n,))
X = jr.normal(data_key, (1000, n))

original = OriginalSPDMatrix(n, N, key=model_key)
original(x)
alternative = AlternativeSPDMatrix(n, N, key=model_key)
alternative(x)


time_code(lambda: original(x), "Original   : No JIT, no vmap: ")
time_code(lambda: alternative(x), "Alternative: No JIT, no vmap: ")

_original = jax.jit(lambda x: original(x))
_original(x)
_alternative = jax.jit(lambda x: alternative(x))
_alternative(x)

time_code(lambda: _original(x), "Original   : JIT, no vmap: ")
time_code(lambda: _alternative(x), "Alternative: JIT, no vmap: ")

_original = jax.vmap(lambda x: original(x))
_original(X)
_alternative = jax.vmap(lambda x: alternative(x))
_alternative(X)

time_code(lambda: _original(X), "Original   : No JIT, vmap: ")
time_code(lambda: _alternative(X), "Alternative: No JIT, vmap: ")