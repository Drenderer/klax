from collections.abc import Callable
from typing import (
    Literal,
    Optional,
    Union,
)

import equinox as eqx
import jax
from jax.nn.initializers import Initializer, zeros
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray

from .._misc import default_floating_dtype
from ._linear import Linear


class MLP(eqx.Module, strict=True):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network.
    
    
    This class is modified form `eqx.nn.MLP` to allow for custom initialization.
    """

    layers: tuple[Linear, ...]
    activation: Callable
    final_activation: Callable
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    in_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int,
        depth: int,
        weight_init: Initializer,
        bias_init: Initializer = zeros,
        activation: Callable = jax.nn.relu,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments**:

        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `out_size`: The output size. The output from the module will be a vector
            of shape `(out_features,)`.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers, including the output layer.
            For example, `depth=2` results in an network with layers:
            [`Linear(in_size, width_size)`, `Linear(width_size, width_size)`,
            `Linear(width_size, out_size)`].
        - `weight_init`: The weight initializer of type `jax.nn.initializers.Initializer`.
        - `bias_init`: The bias initializer of type `jax.nn.initializers.Initializer`.
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `use_bias`: Whether to add on a bias to internal layers. Defaults
            to `True`.
        - `use_final_bias`: Whether to add on a bias to the final layer. Defaults
            to `True`.
        - `dtype`: The dtype to use for all the weights and biases in this MLP.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_size` also supports the string `"scalar"` as a special value.
        In this case the input to the module should be of shape `()`.

        Likewise `out_size` can also be a string `"scalar"`, in which case the
        output from the module will have shape `()`.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        keys = jrandom.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(
                Linear(
                    in_size,
                    out_size,
                    weight_init,
                    bias_init,
                    use_final_bias,
                    dtype=dtype,
                    key=keys[0]
                )
            )
        else:
            layers.append(
                Linear(
                    in_size,
                    width_size,
                    weight_init,
                    bias_init,
                    use_bias,
                    dtype=dtype,
                    key=keys[0]
                )
            )
            for i in range(depth - 1):
                layers.append(
                    Linear(
                        width_size,
                        width_size,
                        weight_init,
                        bias_init,
                        use_bias,
                        dtype=dtype,
                        key=keys[i + 1]
                    )
                )
            layers.append(
                Linear(
                    width_size,
                    out_size,
                    weight_init,
                    bias_init,
                    use_final_bias,
                    dtype=dtype,
                    key=keys[-1]
                )
            )
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        # In case `activation` or `final_activation` are learnt, then make a separate
        # copy of their weights for every neuron.
        self.activation = eqx.filter_vmap(
            eqx.filter_vmap(lambda: activation, axis_size=width_size), axis_size=depth
        )()
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = eqx.filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`. (Or shape `()` if
            `in_size="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`. (Or shape `()` if `out_size="scalar"`.)
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            layer_activation = jtu.tree_map(
                lambda x: x[i] if eqx.is_array(x) else x, self.activation
            )
            x = eqx.filter_vmap(lambda a, b: a(b))(layer_activation, x)
        x = self.layers[-1](x)
        if self.out_size == "scalar":
            x = self.final_activation(x)
        else:
            x = eqx.filter_vmap(lambda a, b: a(b))(self.final_activation, x)
        return x