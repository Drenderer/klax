# Copyright 2025 The Klax Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Historical benchmark: legacy vs. current matrix-valued functions.

This script documents why `klax.nn.SPDMatrix` and
`klax.nn.SkewSymmetricMatrix` are implemented the way they are.

LEGACY implementations (defined in this script, no longer in klax):
    The MLP predicts only the independent matrix entries
    (`n(n+1)/2` for SPD, the strictly lower triangle for skew-symmetric).
    A constant, non-trainable index tensor `_tensor` then scatters these
    entries into the matrix via `einsum`.

CURRENT implementations (imported from `klax.nn`):
    The MLP predicts all `n*n` entries, which are simply reshaped into a
    matrix `B`. The structure is then imposed algebraically:
      * `SPDMatrix`:           `B @ B* + epsilon * I`   (adopted in 174ce5d)
      * `SkewSymmetricMatrix`: `B - B.T`                (adopted in c4142d9)

The current approach slightly over-parameterizes the MLP's output layer, but
avoids the index tensor and the `einsum`, which turned out to be both simpler
and faster. Run this script to reproduce the comparison.
"""

# %% Imports
import timeit
from collections.abc import Callable, Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn.initializers import Initializer, he_normal, zeros
from jaxtyping import Array, PRNGKeyArray

from klax.nn import MLP, SkewSymmetricMatrix, SPDMatrix


# %% Custom code for timeing code
def format_time(seconds: float | Array) -> str:
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


# %% LEGACY implementations (superseded, kept for reference)

type AtLeast2DTuple[T] = tuple[T, T, *tuple[T, ...]]


class LegacySPDMatrix(eqx.Module):
    """LEGACY SPD matrix (superseded by `klax.nn.SPDMatrix` in 174ce5d).

    The MLP predicts the `n(n+1)/2` entries of a lower triangular matrix `L`,
    which are scattered into `L` via a constant index tensor. The output is
    `L @ L* + epsilon * I`.
    """

    func: MLP
    shape: AtLeast2DTuple[int] = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    _tensor: Array  # Not learnable transform tensor from the vector-space of components to the space of lower triangular matrices

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        shape: int | AtLeast2DTuple[int],
        width_sizes: Sequence[int],
        epsilon: float = 1e-6,
        weight_init: Initializer = he_normal(),
        bias_init: Initializer = zeros,  # type: ignore
        activation: Callable = jax.nn.softplus,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype: Any | None = None,
        *,
        key: PRNGKeyArray,
    ):
        shape = shape if isinstance(shape, tuple) else (shape, shape)
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
        l_matrix = jnp.einsum(
            "ijk,...k->...ij", jax.lax.stop_gradient(self._tensor), elements
        )
        a_matrix = l_matrix @ jnp.conjugate(l_matrix.mT)
        identity = jnp.broadcast_to(jnp.eye(self.shape[-1]), a_matrix.shape)
        return a_matrix + self.epsilon * identity


class LegacySkewSymmetricMatrix(eqx.Module):
    """LEGACY skew-symmetric matrix (superseded by `klax.nn.SkewSymmetricMatrix` in c4142d9).

    The MLP predicts the entries of the strictly lower triangle, which are
    scattered into the matrix with `+1`/`-1` signs via a constant index tensor.

    Note:
        Kept verbatim. The output size is `n(n+1)/2` although only the
        `n(n-1)/2` strictly lower entries are used, so `n` MLP outputs are
        wasted.

    """

    func: MLP
    shape: AtLeast2DTuple[int] = eqx.field(static=True)
    _tensor: Array  # Not learnable transform tensor from the vector-space of components to the space of skew-symmetric matrices

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        shape: int | AtLeast2DTuple[int],
        width_sizes: Sequence[int],
        weight_init: Initializer = he_normal(),
        bias_init: Initializer = zeros,  # type: ignore
        activation: Callable = jax.nn.softplus,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype: Any | None = None,
        *,
        key: PRNGKeyArray,
    ):
        shape = shape if isinstance(shape, tuple) else (shape, shape)
        if shape[-1] != shape[-2]:
            raise ValueError(
                "The last two dimensions in shape must be equal for skew-symmetric matrices."
            )

        # Construct the tensor
        n = shape[-1]
        num_batches = int(jnp.prod(jnp.array(shape[:-2])))
        num_elements = n * (n + 1) // 2
        tensor = jnp.zeros((n, n, num_elements), dtype="int32")
        for e, (i, j) in enumerate(zip(*jnp.tril_indices(n, k=-1))):
            tensor = tensor.at[i, j, e].set(1)
            tensor = tensor.at[j, i, e].set(-1)

        self._tensor = tensor
        self.shape = shape
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
        return jnp.einsum(
            "...ijk,...k->...ij", jax.lax.stop_gradient(self._tensor), elements
        )


# %% Benchmark helper


def compare(
    legacy: Callable, current: Callable, x_single: Array, x_batch: Array
) -> None:
    """Time the legacy and current implementation with/without JIT and vmap."""
    variants = {
        "No JIT, no vmap": (lambda f: f, x_single),
        "JIT,    no vmap": (jax.jit, x_single),
        "No JIT, vmap   ": (jax.vmap, x_batch),
        "JIT,    vmap   ": (lambda f: jax.jit(jax.vmap(f)), x_batch),
    }
    for label, (transform, inp) in variants.items():
        _legacy = transform(lambda z: legacy(z))
        _current = transform(lambda z: current(z))
        _legacy(inp)  # Warm up / compile
        _current(inp)
        time_code(lambda: _legacy(inp), f"Legacy  : {label}: ")
        time_code(lambda: _current(inp), f"Current : {label}: ")


# %% Time the different versions

key = jr.key(0)
model_key, data_key = jr.split(key)

n = 50
N = 50

x = jr.normal(data_key, (n,))
X = jr.normal(data_key, (1000, n))

print("\n=== Skew-symmetric matrix ===")
compare(
    LegacySkewSymmetricMatrix(n, N, width_sizes=[128, 128], key=model_key),
    SkewSymmetricMatrix(n, N, width_sizes=[128, 128], key=model_key),
    x,
    X,
)

print("\n=== SPD matrix ===")
compare(
    LegacySPDMatrix(n, N, width_sizes=[128, 128], key=model_key),
    SPDMatrix(n, N, width_sizes=[128, 128], key=model_key),
    x,
    X,
)
