"""
Implementation of convex neural networks.
"""

from __future__ import annotations
from collections.abc import Callable, Sequence
from typing import (
    Literal,
    Optional,
    Union,
)

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, zeros, he_normal
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .._misc import default_floating_dtype
from ..wrappers import NonNegative
from ._linear import Linear, InputSplitLinear


class FICNN(eqx.Module, strict=True):
    """A fully input convex neural network.

    It is convex in its first argument and arbitrary
    in any following argument."""

    layers: tuple[Linear | InputSplitLinear, ...]
    activations: tuple[Callable, ...]
    final_activation: Callable
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    use_passthrough: bool = eqx.field(static=True)
    non_decreasing_first_layer: bool = eqx.field(static=True)
    in_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    width_sizes: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_sizes: Sequence[int],
        use_passthrough: bool = True,
        non_decreasing_first_layer: bool = True,
        weight_init: Initializer = he_normal(),
        bias_init: Initializer = zeros,
        activation: Callable = jax.nn.softplus,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """
        TODO
        """

        def final_activation(x):
            return x

        dtype = default_floating_dtype() if dtype is None else dtype
        width_sizes = tuple(width_sizes)

        self.in_size = in_size
        self.out_size = out_size
        self.width_sizes = width_sizes
        # self.variant = variant
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias
        self.use_passthrough = use_passthrough
        self.non_decreasing_first_layer = non_decreasing_first_layer

        in_sizes = (in_size,) + width_sizes
        out_sizes = width_sizes + (out_size,)
        use_biases = len(width_sizes) * (use_bias,) + (use_final_bias,)
        keys = jrandom.split(key, len(in_sizes))

        layers = []
        for n, (sin, sout, ub, key) in enumerate(
            zip(in_sizes, out_sizes, use_biases, keys)
        ):
            if n == 0:
                layers.append(
                    Linear(
                        sin,
                        sout,
                        weight_init,
                        bias_init,
                        ub,
                        NonNegative if non_decreasing_first_layer else None,
                        dtype=dtype,
                        key=key,
                    )
                )
            else:
                if use_passthrough:
                    layers.append(
                        InputSplitLinear(
                            (sin, in_size),
                            sout,
                            weight_init,
                            bias_init,
                            ub,
                            (NonNegative, None),
                            dtype=dtype,
                            key=key,
                        )
                    )
                else:
                    layers.append(
                        Linear(
                            sin,
                            sout,
                            weight_init,
                            bias_init,
                            ub,
                            NonNegative,
                            dtype=dtype,
                            key=key,
                        )
                    )

        self.layers = tuple(layers)

        # In case `activation` or `final_activation` are learnt, then make a separate
        # copy of their weights for every neuron.
        activations = []
        for width in width_sizes:
            activations.append(eqx.filter_vmap(lambda: activation, axis_size=width)())
        self.activations = tuple(activations)
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = eqx.filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
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
        y = jnp.copy(x)

        for i, (layer, activation) in enumerate(
            zip(self.layers[:-1], self.activations)
        ):
            if i == 0 or not self.use_passthrough:
                y = layer(y)
            else:
                assert isinstance(layer, InputSplitLinear)
                y = layer(y, x)
            layer_activation = jax.tree.map(
                lambda y: y[i] if eqx.is_array(y) else y, activation
            )
            y = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, y)

        if self.use_passthrough:
            assert isinstance(self.layers[-1], InputSplitLinear)
            y = self.layers[-1](y, x)
        else:
            y = self.layers[-1](y)

        if self.out_size == "scalar":
            y = self.final_activation(y)
        else:
            y = eqx.filter_vmap(lambda a, b: a(b))(self.final_activation, y)

        return y
