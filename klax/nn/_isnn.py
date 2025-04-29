"""
Implementations of Input Specific Neural Networks according to
https://arxiv.org/abs/2503.00268
"""

from typing import Callable, Literal, Optional, Union

import equinox as eqx
import jax
from jax.nn.initializers import Initializer, he_normal, zeros
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from ._linear import Linear, InputSplitLinear
from .._misc import default_floating_dtype
from ..wrappers import NonNegative


class ISNN1(eqx.Module, strict=True):
    """ISNN Type 1

    #TODO: Add mathematical descrioption of ISNN here
    """

    c_layers: tuple[InputSplitLinear, ...]
    mc_layers: tuple[Linear, ...]
    m_layers: tuple[Linear, ...]
    a_layers: tuple[Linear, ...]
    c_activation: Callable
    mc_activation: Callable
    m_activation: Callable
    a_activation: Callable
    final_activation: Callable
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    in_sizes: tuple[Union[int, Literal["scalar"]], ...] = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)
    width_sizes: tuple[int, ...] = eqx.field(static=True)
    depths: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        in_sizes: tuple[Union[int, Literal["scalar"]], ...],
        width_sizes: tuple[int, ...],
        depths: tuple[int, ...],
        weight_init: Initializer = he_normal(),
        bias_init: Initializer = zeros,
        a_activation: Callable = jax.nn.relu,
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            in_size: The input size. The input to the module should be a vector
                of shape `(in_features,)`
            out_size: The output size. The output from the module will be a
                vector of shape `(out_features,)`.
            width_sizes: The sizes of each hidden layer in a list.
            weight_init: The weight initializer of type `jax.nn.initializers.Initializer`.
            bias_init: The bias initializer of type `jax.nn.initializers.Initializer`.
            activation: The activation function after each hidden layer.
                (Defaults to ReLU).
            final_activation: The activation function after the output layer.
                (Defaults to the identity.)
            use_bias: Whether to add on a bias to internal layers.
                (Defaults to `True`.)
            use_final_bias: Whether to add on a bias to the final layer.
                (Defaults to `True`.)
            dtype: The dtype to use for all the weights and biases in this MLP.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)

        Note:
            Note that `in_size` also supports the string `"scalar"` as a special
            value. In this case the input to the module should be of shape `()`.

            Likewise `out_size` can also be a string `"scalar"`, in which case
            the output from the module will have shape `()`.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        out_size = "scalar"
        use_final_bias = False

        def final_activation(x):
            return x

        # c_activation = jax.nn.softplus
        # m_activation = jax.nn.tanh

        # Assert shapes
        assert len(in_sizes) == 4, f"Expected len(in_sizes) == 4 but is {len(in_sizes)}"
        assert len(width_sizes) == 4, (
            f"Expected len(width_sizes) == 4 but is {len(width_sizes)}"
        )
        assert len(depths) == 4, f"Expected len(width_sizes) == 4 but is {len(depths)}"

        self.in_sizes = in_sizes
        self.out_size = out_size
        self.width_sizes = width_sizes
        self.depths = depths
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

        # Monotonically non-decreasing and convex branch
        mc_in_sizes = (in_sizes[1],) + (width_sizes[1],) * (depths[1] - 1)
        mc_out_sizes = (width_sizes[1],) * depths[1]
        mc_keys = jrandom.split(key, depths[1])

        self.mc_layers = tuple(
            Linear(
                sin,
                sout,
                weight_init,
                bias_init,
                use_bias,
                NonNegative,
                dtype=dtype,
                key=key,
            )
            for sin, sout, key in zip(mc_in_sizes, mc_out_sizes, mc_keys)
        )

        # Monotonically non-decreasing branch
        m_in_sizes = (in_sizes[2],) + (width_sizes[2],) * (depths[2] - 1)
        m_out_sizes = (width_sizes[2],) * depths[2]
        m_keys = jrandom.split(key, depths[2])

        self.m_layers = tuple(
            Linear(
                sin,
                sout,
                weight_init,
                bias_init,
                use_bias,
                NonNegative,
                dtype=dtype,
                key=key,
            )
            for sin, sout, key in zip(m_in_sizes, m_out_sizes, m_keys)
        )

        # Arbitrary branch
        a_in_sizes = (in_sizes[3],) + (width_sizes[3],) * (depths[3] - 1)
        a_out_sizes = (width_sizes[3],) * depths[3]
        a_keys = jrandom.split(key, depths[3])

        self.a_layers = tuple(
            Linear(sin, sout, weight_init, bias_init, use_bias, dtype=dtype, key=key)
            for sin, sout, key in zip(a_in_sizes, a_out_sizes, a_keys)
        )

        # Convex branch
        c_layers = []
        for n in range(depths[0]):
            if n == 0:
                c_layers.append(
                    InputSplitLinear(
                        (
                            in_sizes[0],
                            width_sizes[1],
                            width_sizes[2],
                            width_sizes[3],
                        ),
                        width_sizes[0],
                        weight_init,
                        bias_init,
                        use_bias,
                        (None, NonNegative, NonNegative, None),
                        dtype=dtype,
                        key=key,
                    )
                )
            else:
                c_layers.append(
                    InputSplitLinear(
                        (in_sizes[0], width_sizes[0]),
                        width_sizes[0],
                        weight_init,
                        bias_init,
                        use_bias,
                        (None, NonNegative),
                        dtype=dtype,
                        key=key,
                    )
                )
        c_layers.append(
            InputSplitLinear(
                (in_sizes[0], width_sizes[0]),
                out_size,
                weight_init,
                bias_init,
                use_bias,
                (None, NonNegative),
                dtype=dtype,
                key=key,
            )
        )
        self.c_layers = tuple(c_layers)

        # In case `activation` or `final_activation` are learnt, then make a separate
        # copy of their weights for every neuron.
        self.c_activation = eqx.filter_vmap(
            eqx.filter_vmap(lambda: jax.nn.softplus, axis_size=width_sizes[0]),
            axis_size=depths[0],
        )()
        self.mc_activation = eqx.filter_vmap(
            eqx.filter_vmap(lambda: jax.nn.softplus, axis_size=width_sizes[1]),
            axis_size=depths[1],
        )()
        self.m_activation = eqx.filter_vmap(
            eqx.filter_vmap(lambda: jax.nn.softplus, axis_size=width_sizes[2]),
            axis_size=depths[2],
        )()
        self.a_activation = eqx.filter_vmap(
            eqx.filter_vmap(lambda: a_activation, axis_size=width_sizes[3]),
            axis_size=depths[3],
        )()

        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = eqx.filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()

    def __call__(
        self,
        *xs: Array,
        key: Optional[PRNGKeyArray] = None,
    ) -> Array:
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
        x, y, t, z = xs

        for i, layer in enumerate(self.mc_layers):
            y = layer(y)
            layer_activation = jax.tree.map(
                lambda x: x[i] if eqx.is_array(x) else x, self.mc_activation
            )
            y = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, y)

        for i, layer in enumerate(self.m_layers):
            t = layer(t)
            layer_activation = jax.tree.map(
                lambda t: t[i] if eqx.is_array(t) else t, self.m_activation
            )
            t = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, t)

        for i, layer in enumerate(self.a_layers):
            z = layer(z)
            layer_activation = jax.tree.map(
                lambda z: z[i] if eqx.is_array(z) else z, self.a_activation
            )
            z = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, z)

        x0 = jnp.copy(x)
        for i, layer in enumerate(self.c_layers[:-1]):
            if i == 0:
                x = layer(x0, y, t, z)
            else:
                x = layer(x0, x)
            layer_activation = jax.tree.map(
                lambda x: x[i] if eqx.is_array(x) else x, self.c_activation
            )
            x = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, x)
        x = self.c_layers[-1](x0, x)
        if self.out_size == "scalar":
            x = self.final_activation(x)
        else:
            x = eqx.filter_vmap(lambda a, b: a(b))(self.final_activation, x)
        return x
