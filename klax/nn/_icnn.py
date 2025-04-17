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
from jax.nn.initializers import Initializer, zeros, he_normal
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .._misc import default_floating_dtype
from ..wrappers import NonNegative
from ._linear import Linear, InputSplitLinear


class FICNN(eqx.Module, strict=True):
    """
    A [Fully Input Convex Neural Network](https://arxiv.org/abs/1609.07152). 
    Each element of the output is a convex function of the input.
    """

    layers: tuple[Linear | InputSplitLinear, ...]
    activations: tuple[Callable, ...]
    final_activation: Callable
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    use_passthrough: bool = eqx.field(static=True)
    non_decreasing: bool = eqx.field(static=True)
    in_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    width_sizes: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_sizes: Sequence[int],
        use_passthrough: bool = True,
        non_decreasing: bool = False,
        weight_init: Initializer = he_normal(),
        bias_init: Initializer = zeros,
        activation: Callable = jax.nn.softplus,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            in_size: The input size. The input to the module should be a vector
                of shape `(in_features,)`.
            out_size: The output size. The output from the module will be a
                vector of shape `(out_features,)`.
            width_sizes: The sizes of each hidden layer in a list.
            use_passthrough: Whether to use passthrough layers. If true, the input
             is passed through to each hidden layer. Defaults to True.
            non_decreasing: If true, the output is element-wise non-decreasing 
                in each input. This is useful if the input `x` is a convex function 
                of some other quantity `z`. If the FICNN `f(x(z))` is non-decreasing 
                then f preserves the convexity with respect to `z`. Defaults to False.
            weight_init: The weight initializer of type `jax.nn.initializers.Initializer`.
                Defaults to he_normal().
            bias_init: The bias initializer of type `jax.nn.initializers.Initializer`.
                Defaults to zeros.
            activation: The activation function of each hidden layer. To ensure 
                convexity this function must be convex and non-decreasing. 
                Defaults to `jax.nn.softplus`.
            final_activation: The activation function after the output layer. To ensure 
                convexity this function must be convex and non-decreasing. 
                Defaults to the identity.
            use_bias: Whether to add on a bias in the hidden layers. Defaults to True.
            use_final_bias: Whether to add on a bias to the final layer. Defaults to True.
            dtype: The dtype to use for all the weights and biases in this MLP.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)
        """

        # What's up with this? Why not let the user define a final activation?
        # def final_activation(x):  
        #     return x

        dtype = default_floating_dtype() if dtype is None else dtype
        width_sizes = tuple(width_sizes)

        self.in_size = in_size
        self.out_size = out_size
        self.width_sizes = width_sizes
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias
        self.use_passthrough = use_passthrough
        self.non_decreasing = non_decreasing

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
                        NonNegative if non_decreasing else None,
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
                            (NonNegative, NonNegative) if non_decreasing else (NonNegative, None),
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

        y = self.layers[0](x)

        for i, (layer, activation) in enumerate(
            zip(self.layers[1:], self.activations)
        ):
            layer_activation = jax.tree.map(
                lambda y: y[i] if eqx.is_array(y) else y, activation
            )
            y = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, y)

            if self.use_passthrough:
                assert isinstance(layer, InputSplitLinear)
                y = layer(y, x)
            else:
                y = layer(y)

        if self.out_size == "scalar":
            y = self.final_activation(y)
        else:
            y = eqx.filter_vmap(lambda a, b: a(b))(self.final_activation, y)

        return y
