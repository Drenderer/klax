"""Comparison of SPDMatrix to an alternative implementation."""

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

from klax.nn import MLP


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


# %% Define implementations

type AtLeast2DTuple[T] = tuple[T, T, *tuple[T, ...]]


class OriginalSPDMatrix(eqx.Module):
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


class AlternativeSPDMatrix(eqx.Module):
    func: MLP
    shape: AtLeast2DTuple[int] = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

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
        l_matrix = self.func(x).reshape(self.shape)
        a_matrix = l_matrix @ jnp.conjugate(l_matrix.mT)
        identity = jnp.broadcast_to(jnp.eye(self.shape[-1]), a_matrix.shape)
        return a_matrix + self.epsilon * identity


class OriginalSkewSymmetricMatrix(eqx.Module):
    """*Skew-symmetric matrix-valued function*.

    Wrapper around a_matrix `MLP` that maps the input to a_matrix vector of elements that
    are transformed to a_matrix skew-symmetric matrix.
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
        """Initialize the `SkewSymmetricMatrix`.

        Args:
            in_size: The input size. The input to the module should be a_matrix vector
                of shape `(in_size,)`
            shape: The matrix shape. The output from the module will be a_matrix
                Array with sthe specified `shape`. For square matrices a_matrix single
                integer N can be used as a_matrix shorthand for (N, N).
                (Defaults to `(in_size, in_size)`)
            width_sizes: The sizes of each hidden layer of the underlying MLP in a_matrix list.
                (Defaults to `[k,]`, where `k` is the smallest power of 2 greater or
                equal to `in_size`.)
            weight_init: The weight initializer of type `jax.nn.initializers.Initializer`.
                (Defaults to `he_normal()`)
            bias_init: The bias initializer of type `jax.nn.initializers.Initializer`.
                (Defaults to `zeros`)
            activation: The activation function after each hidden layer.
                (Defaults to `softplus`).
            final_activation: The activation function after the output layer.
                (Defaults to the identity.)
            use_bias: Whether to add on a_matrix bias to internal layers.
                (Defaults to `True`.)
            use_final_bias: Whether to add on a_matrix bias to the final layer.
                (Defaults to `True`.)
            dtype: The dtype to use for all the weights and biases in this MLP.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.
            key: a_matrix `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)

        Note:
            Note that `in_size` also supports the string `"scalar"` as a_matrix special
            value. In this case the input to the module should be of shape `()`.

        """
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


class AlternativeSkewSymmetricMatrix(eqx.Module):
    """*Skew-symmetric matrix-valued function*.

    Wrapper around a_matrix `MLP` that maps the input to a_matrix vector of elements that
    are transformed to a_matrix skew-symmetric matrix.
    """

    func: MLP
    shape: AtLeast2DTuple[int] = eqx.field(static=True)

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
        """Initialize the `SkewSymmetricMatrix`.

        Args:
            in_size: The input size. The input to the module should be a_matrix vector
                of shape `(in_size,)`
            shape: The matrix shape. The output from the module will be a_matrix
                Array with sthe specified `shape`. For square matrices a_matrix single
                integer N can be used as a_matrix shorthand for (N, N).
                (Defaults to `(in_size, in_size)`)
            width_sizes: The sizes of each hidden layer of the underlying MLP in a_matrix list.
                (Defaults to `[k,]`, where `k` is the smallest power of 2 greater or
                equal to `in_size`.)
            weight_init: The weight initializer of type `jax.nn.initializers.Initializer`.
                (Defaults to `he_normal()`)
            bias_init: The bias initializer of type `jax.nn.initializers.Initializer`.
                (Defaults to `zeros`)
            activation: The activation function after each hidden layer.
                (Defaults to `softplus`).
            final_activation: The activation function after the output layer.
                (Defaults to the identity.)
            use_bias: Whether to add on a_matrix bias to internal layers.
                (Defaults to `True`.)
            use_final_bias: Whether to add on a_matrix bias to the final layer.
                (Defaults to `True`.)
            dtype: The dtype to use for all the weights and biases in this MLP.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.
            key: a_matrix `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)

        Note:
            Note that `in_size` also supports the string `"scalar"` as a_matrix special
            value. In this case the input to the module should be of shape `()`.

        """
        shape = shape if isinstance(shape, tuple) else (shape, shape)
        if shape[-1] != shape[-2]:
            raise ValueError(
                "The last two dimensions in shape must be equal for skew-symmetric matrices."
            )

        num_elements = int(jnp.prod(jnp.array(shape)))
        self.shape = shape
        self.func = MLP(
            in_size,
            num_elements,
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
        a_matrix = self.func(x).reshape(self.shape)
        return a_matrix - a_matrix.mT


# %% Time the different versions

key = jr.key(0)
model_key, data_key = jr.split(key)

n = 50
N = 50

x = jr.normal(data_key, (n,))
X = jr.normal(data_key, (1000, n))

original = OriginalSkewSymmetricMatrix(
    n, N, width_sizes=[128, 128], key=model_key
)
original(x)
alternative = AlternativeSkewSymmetricMatrix(
    n, N, width_sizes=[128, 128], key=model_key
)
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

_original = jax.jit(jax.vmap(lambda x: original(x)))
_original(X)
_alternative = jax.jit(jax.vmap(lambda x: alternative(x)))
_alternative(X)

time_code(lambda: _original(X), "Original   : JIT, vmap: ")
time_code(lambda: _alternative(X), "Alternative: JIT, vmap: ")
