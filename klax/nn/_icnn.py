# Copyright 2025 The Klax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementations of convex neural networks."""

from collections.abc import Callable, Sequence
from typing import Literal, cast

import equinox as eqx
import jax
import jax.random as jr
from jax.nn.initializers import he_normal, ones, zeros
from jaxtyping import Array, Float, PRNGKeyArray

from .._initializers import SupportedInitializer, hoedt_bias, hoedt_normal
from .._misc import default_floating_dtype
from .._wrappers import NonNegative
from ._linear import InputSplitLinear, Linear


class FICNN(eqx.Module, strict=True):
    """A fully input convex neural network (FICNN) according to [Amos et al.](https://arxiv.org/abs/1609.07152).

    Each element of the output `y` is a convex function of the input `x`.

    """

    layers: tuple[Linear | InputSplitLinear, ...]
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
        weight_init: SupportedInitializer = he_normal(),
        bias_init: SupportedInitializer = zeros,
        constrained_weight_init: SupportedInitializer | None = hoedt_normal(),
        constrained_bias_init: SupportedInitializer | None = hoedt_bias(),
        activation: Callable = jax.nn.softplus,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype: type | None = None,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize FICNN.

        Warning:
            Modifying `activation` or `final_activation` to a concave function
            or a function that isn't non-decreasing will break the convexity of
            the FICNN. Use these parameters with care.

        Args:
            in_size: The input size. The input to the module should be a vector
                of shape `(in_size,)`.
            out_size: The output size. The output from the module will be a
                vector of shape `(out_size,)`.
            width_sizes: The sizes of each hidden layer provided as a list.
            use_passthrough: Whether to use passthrough layers. If true, the
                input is passed through to each hidden layer and the output
                layer. Defaults to `True`.
            non_decreasing: If true, all weights in the first layer are
                constrained using `klax.NonNegative`. Hence, the output is
                element-wise non-decreasing in each input. This is useful in
                the following scenario: Consider that you want to model the
                function `g(z) = FICNN(x(z))` as a chain of the functions
                `x(z)` and `FICNN(x)` such that `g` is convex w.r.t.
                the `z`. If `x(z)` is convex, **the FICNN must be
                non-decreasing** in `x` to ensure convexity of `g(z)`. This
                option is, for example, used in material modeling applications,
                where the FICNN is a function of convex invariants, c.f., [Dammaß et al. (2025)](https://doi.org/10.48550/arXiv.2503.20598).
            weight_init: The weight initializer of type `SupportedInitializer`
                used for *unconstrained weights*.
                Defaults to he_normal().
            bias_init: The bias initializer of type `SupportedInitializer` used
                for the biases of *unconstrained layers*.
                Defaults to zeros.
            constrained_weight_init: The weight initializer of type
                `SupportedInitializer` used for *constrained weights*.
                If None, then `weight_init` is used for constrained weights as well.
                Defaults to [`klax.hoedt_normal`][].
            constrained_bias_init: The bias initializer of type
                `SupportedInitializer` used for the biases of *constrained layers*.
                If None, then `bias_init` is used for the biases in constrained
                layers as well.
                Defaults to zeros.
            activation: The activation function of each hidden layer. To ensure
                convexity this function must be convex and non-decreasing.
                Defaults to `jax.nn.softplus`.
            final_activation: The activation function after the output layer.
                To ensure convexity this function must be convex and
                non-decreasing. (Defaults to the identity.)
            use_bias: Whether to add on a bias in the hidden layers. (Defaults
                to True.)
            use_final_bias: Whether to add on a bias to the final layer.
                Defaults to True.
            dtype: The dtype to use for all the weights and biases in this MLP.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.
            key: A `jax.random.PRNGKey` used to provide randomness for
                parameter initialization. (Keyword only argument.)

        """
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
        keys = jr.split(key, len(in_sizes))

        constrained_weight_init = (
            weight_init
            if constrained_weight_init is None
            else constrained_weight_init
        )
        constrained_bias_init = (
            bias_init
            if constrained_bias_init is None
            else constrained_bias_init
        )

        layers = []
        for n, (sin, sout, ub, key) in enumerate(
            zip(in_sizes, out_sizes, use_biases, keys)
        ):
            if n == 0:
                layers.append(
                    Linear(
                        sin,
                        sout,
                        (
                            constrained_weight_init
                            if non_decreasing
                            else weight_init
                        ),
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
                            (
                                constrained_weight_init
                                if non_decreasing
                                else (constrained_weight_init, weight_init)
                            ),
                            bias_init,
                            ub,
                            (
                                (NonNegative, NonNegative)
                                if non_decreasing
                                else (NonNegative, None)
                            ),
                            dtype=dtype,
                            key=key,
                        )
                    )
                else:
                    layers.append(
                        Linear(
                            sin,
                            sout,
                            constrained_weight_init,
                            constrained_bias_init,
                            ub,
                            NonNegative,
                            dtype=dtype,
                            key=key,
                        )
                    )

        self.layers = tuple(layers)

        # In case `activation` or `final_activation` are learnt, then make a
        # separate copy of their weights for every neuron.
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
        """Forward pass through `FICNN`.

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
                # Tell type checker that this is an InputSplitLinear
                layer = cast(InputSplitLinear, layer)
                y = layer(y, x)
            else:
                y = layer(y)

        if self.out_size == "scalar":
            y = self.final_activation(y)
        else:
            y = eqx.filter_vmap(lambda a, b: a(b))(self.final_activation, y)

        return y


class PICNNLayer(eqx.Module, strict=True):
    """Layer for a partially input convex neural network from [Amos et al.](https://arxiv.org/abs/1609.07152)."""

    linear_y: InputSplitLinear
    linear_u: Linear | None
    linear_yu: Linear
    linear_xu: Linear | None
    activation_y: Callable
    activation_u: Callable | None
    activation_yu: Callable
    activation_xu: Callable | None
    use_bias: bool = eqx.field(static=True)
    use_passthrough: bool = eqx.field(static=True)
    nonnegative_y_weight: bool = eqx.field(static=True)
    nonnegative_passthrough: bool = eqx.field(static=True)
    update_u: bool = eqx.field(static=True)

    def __init__(
        self,
        y_in_size: int | Literal["scalar"],
        u_in_size: int | Literal["scalar"],
        x_size: int | Literal["scalar"],
        y_out_size: int | Literal["scalar"],
        u_out_size: int | Literal["scalar"],
        *,
        use_passthrough: bool = True,
        nonnegative_y_weight: bool = True,
        nonnegative_passthrough: bool = False,
        use_bias: bool = True,
        update_u: bool = True,
        activation_y: Callable = jax.nn.softplus,
        activation_u: Callable = jax.nn.softplus,
        activation_yu: Callable = jax.nn.softplus,
        activation_xu: Callable = lambda x: x,
        weight_init: SupportedInitializer = he_normal(),
        bias_init: SupportedInitializer = zeros,
        constrained_weight_init: SupportedInitializer | None = hoedt_normal(),
        constrained_bias_init: SupportedInitializer | None = hoedt_bias(),
        interconnect_weight_init: SupportedInitializer | None = zeros,
        interconnect_bias_init: SupportedInitializer | None = ones,
        dtype: type | None = None,
        key: PRNGKeyArray,
    ):
        """Initialize the PICNN layer.

        Info: Explanation of the variable names and intuition.
            A PICNN is essentially a FICNN, where the weight matrices
            and the bias are functions of an additional input `p`.
            This way, the output is convex in the original input `x`,
            but has an arbitrary relationship to the input `p` - the
            input `p` modulates the network.

            The input `p` is passed into an MLP, that we call the
            "arbitrary path", since it is the flow of information
            through the network without any constraints. Information
            about the input `x` never enters the "arbitrary path".
            The information path from the input `x` to the output is
            called the "convex path". Together with the "passthrough
            path" it essentially constitutes a FICNN.
            Information from the hidden layer activation `u` in the
            arbitrary path flows to the hidden layer activations `y`
            of the convex path, via "interconnection paths". These are
            essentially simple single layer MLPs that take `u` from the
            arbitrary path as argument and output a modulation vector
            for adapting the weights in the convex path.

        Args:
            y_in_size: Size of the convex path input.
            y_out_size: Size of the convex path output.
            u_in_size: Size of the arbitrary path input.
            u_out_size: Size of the arbitrary path output.
            x_size: Size of the passthrough path input
            use_passthrough: If true, the original input `x` is used in the
                calculation of the output `y`.
                Defaults to True.
            nonnegative_y_weight: If true, applies the [nonnegative][klax.NonNegative]
                weight wrapper to the appropriate weight to ensure convexity of the
                output `y` with respect to the input `y`. However, note that the first
                PICNN layer in a PICNN should not enforce this positivity, unless
                the PICNN output is supposed to be non-decreasing in the convex input
                as well.
                Defaults to True.
            nonnegative_passthrough: If true, applies the [nonnegative][klax.NonNegative]
                weight wrapper to all weights in the passthrough path. This is only
                necessary if the PICNN should be non-decreasing. However, in that case
                you should consider to avoiding passthrough layers, as their benefit
                will be largely diminished.
                Defaults to False.
            use_bias: Whether to use a bias in the convex path. All layers in the
                arbitrary and interconnection paths will use biases regardless of
                this arguments value.
                Defaults to True.
            update_u: If true updates the hidden state `u` using a single layer MLP.
                Note that this should be most likely False in the last layer of an
                FICNN.
                Defaults to True.
            activation_y: Activation applied to the convex path output.
                Defaults to jax.nn.softplus().
            activation_u: Activation applied to the arbitrary path output.
                Defaults to jax.nn.softplus().
            activation_yu: Activation applied to the interconnection weight
                modulation that acts on the convex path weight.
                Defaults to jax.nn.softplus().
            activation_xu: Activation applied to the interconnection weight
                modulation that acts on the passthrough weight.
                Defaults to the identity (`lambda x: x`).
            weight_init: Default weight initialization. Defaults to he_normal().
            bias_init: Default bias initialization. Defaults to zeros.
            constrained_weight_init: The weight initializer used for
                *nonnegative constrained weights*.
                If None, then `weight_init` is used for constrained weights as well.
                Note that if `nonnegative_y_weight=True` then this argument is ignored
                and the default `weight_init` is used instead.
                Defaults to [`klax.hoedt_normal`][].
            constrained_bias_init: The bias initializer used for biases in layers
                with *nonnegative constrained weights*.
                Defaults to hoedt_bias().
            interconnect_weight_init: Weight initializer for weights in the
                interconnection path from the arbitrary path to the convex path.
                Defaults to `zeros`.
            interconnect_bias_init: Bias initializer for weights in the
                interconnection path from the arbitrary path to the convex path.
                Defaults to `ones`.
            dtype: The dtype to use for all the weights and biases in this MLP.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.
            key: A `jax.random.PRNGKey` used to provide randomness for
                parameter initialization.

        """
        dtype = default_floating_dtype() if dtype is None else dtype

        constrained_weight_init = (
            weight_init
            if constrained_weight_init is None
            else constrained_weight_init
        )
        constrained_bias_init = (
            bias_init
            if constrained_bias_init is None
            else constrained_bias_init
        )
        interconnect_weight_init = (
            weight_init
            if interconnect_weight_init is None
            else interconnect_weight_init
        )
        interconnect_bias_init = (
            bias_init
            if interconnect_bias_init is None
            else interconnect_bias_init
        )

        # In case any `activation` is learnt, make a
        # separate copy of their weights for every neuron.
        activation_y = eqx.filter_vmap(
            lambda: activation_y, axis_size=y_out_size
        )()
        activation_u = eqx.filter_vmap(
            lambda: activation_u, axis_size=u_out_size
        )()
        activation_yu = eqx.filter_vmap(
            lambda: activation_yu, axis_size=y_in_size
        )()
        activation_xu = eqx.filter_vmap(
            lambda: activation_xu, axis_size=x_size
        )()

        key_y, key_u, key_yu, key_xu = jr.split(key, 4)
        if use_passthrough:
            if nonnegative_passthrough:
                weight_wraps = (
                    (NonNegative, NonNegative, None)
                    if nonnegative_y_weight
                    else (None, NonNegative, None)
                )
            else:
                weight_wraps = (
                    (NonNegative, None, None) if nonnegative_y_weight else None
                )

            self.linear_y = InputSplitLinear(
                (y_in_size, x_size, u_in_size),
                y_out_size,
                weight_inits=(
                    constrained_weight_init
                    if nonnegative_y_weight
                    else weight_init,
                    constrained_weight_init
                    if nonnegative_passthrough
                    else weight_init,
                    interconnect_weight_init,
                ),
                bias_init=constrained_bias_init
                if nonnegative_y_weight
                else bias_init,
                use_bias=use_bias,
                weight_wraps=weight_wraps,
                dtype=dtype,
                key=key_y,
            )
            self.linear_xu = Linear(
                u_in_size,
                x_size,
                weight_init=interconnect_weight_init,
                bias_init=interconnect_bias_init,
                use_bias=True,
                dtype=dtype,
                key=key_xu,
            )
            self.activation_xu = activation_xu
        else:
            self.linear_y = InputSplitLinear(
                (y_in_size, u_in_size),
                y_out_size,
                weight_inits=(
                    constrained_weight_init
                    if nonnegative_y_weight
                    else weight_init,
                    interconnect_weight_init,
                ),
                bias_init=constrained_bias_init,
                use_bias=use_bias,
                weight_wraps=(NonNegative, None)
                if nonnegative_y_weight
                else None,
                dtype=dtype,
                key=key_y,
            )
            self.linear_xu = None
            self.activation_xu = None
        self.activation_y = activation_y

        if update_u:
            self.linear_u = Linear(
                u_in_size,
                u_out_size,
                weight_init=weight_init,
                bias_init=bias_init,
                use_bias=True,
                dtype=dtype,
                key=key_u,
            )
            self.activation_u = activation_u
        else:
            self.linear_u = None
            self.activation_u = None
        self.linear_yu = Linear(
            u_in_size,
            y_in_size,
            weight_init=interconnect_weight_init,
            bias_init=interconnect_bias_init,
            use_bias=True,
            dtype=dtype,
            key=key_yu,
        )
        self.activation_yu = activation_yu

        self.use_bias = use_bias
        self.use_passthrough = use_passthrough
        self.nonnegative_y_weight = nonnegative_y_weight
        self.nonnegative_passthrough = nonnegative_passthrough
        self.update_u = update_u

    def __call__(
        self,
        y: Float[Array, "... y_in_size"],
        u: Float[Array, "... u_in_size"],
        x: Float[Array, "... x_size"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[
        Float[Array, "... y_out_size"],
        Float[Array, "... u_out_size"],
        Float[Array, "... x_size"],
    ]:
        w_y = self.activation_yu(self.linear_yu(u)) * y
        if self.use_passthrough:
            w_x = self.activation_xu(self.linear_xu(u)) * x
            y = self.activation_y(self.linear_y(w_y, w_x, u))
        else:
            y = self.activation_y(self.linear_y(w_y, u))
        if self.update_u:
            u = self.activation_u(self.linear_u(u))

        return y, u, x


class PICNN(eqx.Module, strict=True):
    """A partially input convex neural network (FICNN) accordin to [Amos et al.](https://arxiv.org/abs/1609.07152).

    A PICNN is a function `(x, p) -> y` where each element
    of the output `y` is a convex function of the input `x`,
    but can have an arbitrary relationship to the input `p`.
    You can think of the PICNN as an FICNN mapping `x` to `y`, but
    whose weights and biases are functions of the input `p`.
    """

    layers: tuple[PICNNLayer]
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    use_passthrough: bool = eqx.field(static=True)
    non_decreasing: bool = eqx.field(static=True)
    x_size: int | Literal["scalar"] = eqx.field(static=True)
    p_size: int | Literal["scalar"] = eqx.field(static=True)
    out_size: int | Literal["scalar"] = eqx.field(static=True)
    width_sizes: Sequence[tuple[int, int]] = eqx.field(static=True)

    def __init__(
        self,
        x_size: int | Literal["scalar"],
        p_size: int | Literal["scalar"],
        out_size: int | Literal["scalar"],
        width_sizes: Sequence[int | tuple[int, int]],
        *,
        use_passthrough: bool = True,
        non_decreasing: bool = False,
        weight_init: SupportedInitializer = he_normal(),
        bias_init: SupportedInitializer = zeros,
        constrained_weight_init: SupportedInitializer | None = hoedt_normal(),
        constrained_bias_init: SupportedInitializer | None = hoedt_bias(),
        interconnect_weight_init: SupportedInitializer | None = zeros,
        interconnect_bias_init: SupportedInitializer | None = ones,
        activation_y: Callable = jax.nn.softplus,
        activation_u: Callable = jax.nn.softplus,
        activation_yu: Callable = jax.nn.softplus,
        activation_xu: Callable = lambda x: x,
        final_activation_y: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype: type | None = None,
        key: PRNGKeyArray,
    ):
        """Initialize PICNN.

        The PICNN's output `y` will be an element-wise convex function of the
        input `x`, but can have an arbitrary functional relationship to the input `p`.

        Warning:
            To ensure convexity, the activation functions need to have certain
            properties, depending on the `non_decreasing` option:

            | Activation | `non_decreasing=False` | `non_decreasing=True` |
            |---|---|---|
            | `activation_y` | *Convex and non-decreasing* | *Convex and non-decreasing* |
            | `final_activation_y` | *Convex and non-decreasing* | *Convex and non-decreasing* |
            | `activation_u` | Arbitrary | Arbitrary |
            | `activation_yu` | *Non-negative* | *Non-negative* |
            | `activation_xu` | Arbitrary | *Non-negative* |

        Args:
            x_size: Size of the input `x`. Can be `"scalar"`, to indicate a scalar input.
                The input to the PICNN should be a vector of shape `(x_size,)` or
                a scalar with shape `()` if `x_size="scalar"`.
            p_size: Size of the input `p`. Can be `"scalar"`, to indicate a scalar input.
                The input to the PICNN should be a vector of shape `(p_size,)` or
                a scalar with shape `()` if `p_size="scalar"`.
            out_size: Size of the output `y`. Can be `"scalar"`, to indicate a scalar output.
                The output of the PICNN will be a vector of shape `(out_size,)` or
                a scalar with shape `()` if `out_size="scalar"`.
            width_sizes: List of the sizes for each hidden layer. Each element of the list
                can be either an integer or a tuple of two integers `(y_size, u_size)`.
                In the latter case the layers output sizes can be defined individually
                for both the convex output `y(x, p)` and the arbitrary output `p(p)`.
                If only a single integer is provided, it is used for both `y_size and u_size`.
            use_passthrough: use_passthrough: Whether to use passthrough layers.
                If true, the PICNN's input is passed again to each hidden layer (Except
                for the first hidden layer, since it receives the original input anyway).
                Defaults to True.
            non_decreasing: If true, the `PICNN` output is element-wise
                non-decreasing in the input `x`. This is useful in the
                following scenario: Consider that you want to model the function
                `g(z, p) = PICNN(x(z), p)` as a chain of the functions `x(z)` and
                `PICNN(x, z)` such that `g` is convex w.r.t. `z`. If `x(z)` is
                convex, **the PICNN must be non-decreasing** in `x` to ensure
                convexity of `g(z)`. This option is, for example, used in
                material modeling applications, where the PICNN is a function
                of convex invariants, c.f., [Dammaß et al. (2025)](https://doi.org/10.48550/arXiv.2503.20598).

                Note: If `use_passthrough=True` and `non_decreasing=True` then
                `activation_xu` has to be a non-negative function to guarantee
                convexity with respect to `x`. We recomment avoiding passthrough
                for non-decreasing PICNNs.
                Defaults to False.
            weight_init: The weight initializer of type `SupportedInitializer`
                used for *unconstrained weights*.
                Defaults to he_normal().
            bias_init: The bias initializer of type `SupportedInitializer`
                used for biases in *sublayers without weight constraints*.
                Defaults to zeros.
            constrained_weight_init: The weight initializer of type `SupportedInitializer`
                used for *constrained weights*.
                Can be `None`, in which case `weight_init` is used.
                Defaults to hoedt_normal().
            constrained_bias_init: The bias initializer of type `SupportedInitializer`
                used for biases in *sublayers with weight constraints*.
                Can be `None`, in which case `bias_init` is used.
                Defaults to hoedt_bias().
            interconnect_weight_init: The weight initializer of type `SupportedInitializer`
                used for weights in *interconnection sublayers*.
                By initializing these weights to zero, the initial PICNN's
                output will not depend on the input `p`.
                Defaults to zeros.
            interconnect_bias_init: The bias initializer of type `SupportedInitializer`
                used for biases in *interconnection sublayers*.
                Defaults to ones.
            activation_y: Activation function applied to the convex output
                `y` in all but the last layer. To ensure convexity, this
                function is required to be *convex and non-decreasing*.
                Defaults to jax.nn.softplus.
            activation_u: Activation function applied to the arbitrary output
                `u`. Defaults to jax.nn.softplus.
            activation_yu: Activation function applied to the output
                of the sublayer computing the weight modulation in the
                convex path. To ensure convexity, this function is required
                to be *non-negative*.
                Defaults to jax.nn.softplus.
            activation_xu: Activation function applied to the output
                of the sublayer computing the weight modulation in the
                passthrough path. To ensure convexity when `non_decreasing=True`,
                this function is required to be *non-negative*, otherwise there
                are no constraints.
                Defaults to `lambdax: x`.
            final_activation_y: Activation function applied to the convex output
                `y` of th last layer. To ensure convexity, this
                function is required to be *convex and non-decreasing*.
                Defaults to `lambdax: x`.
            use_bias: Whether to add on a bias in the hidden layers.
                Defaults to True.
            use_final_bias: Whether to add on a bias to the final layer.
                Defaults to True.
            dtype: The dtype to use for all the weights and biases in this MLP.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.
            key: A `jax.random.PRNGKey` used to provide randomness for
                parameter initialization. (Keyword only argument.)

        """
        dtype = default_floating_dtype() if dtype is None else dtype

        width_sizes = tuple(
            (n, n) if isinstance(n, int) else n for n in width_sizes
        )
        in_sizes = ((x_size, p_size),) + width_sizes
        out_sizes = width_sizes + ((out_size, None),)
        use_passthroughs = (False,) + len(width_sizes) * (use_passthrough,)
        use_biases = len(width_sizes) * (use_bias,) + (use_final_bias,)
        update_u = len(width_sizes) * (True,) + (False,)
        nonnegative_y_weight = (non_decreasing,) + len(width_sizes) * (True,)
        activations_y = len(width_sizes) * (activation_y,) + (
            final_activation_y,
        )
        keys = jr.split(key, len(in_sizes))
        layers = []
        for (in_y, in_u), (out_y, out_u), up, ub, uu, enn, ay, k in zip(
            in_sizes,
            out_sizes,
            use_passthroughs,
            use_biases,
            update_u,
            nonnegative_y_weight,
            activations_y,
            keys,
        ):
            layers.append(
                PICNNLayer(
                    y_in_size=in_y,
                    u_in_size=in_u,
                    x_size=x_size,
                    y_out_size=out_y,
                    u_out_size=out_u,
                    use_passthrough=up,
                    nonnegative_y_weight=enn,
                    nonnegative_passthrough=non_decreasing,
                    use_bias=ub,
                    update_u=uu,
                    activation_y=ay,
                    activation_u=activation_u,
                    activation_yu=activation_yu,
                    activation_xu=activation_xu,
                    weight_init=weight_init,
                    bias_init=bias_init,
                    constrained_weight_init=constrained_weight_init,
                    constrained_bias_init=constrained_bias_init,
                    interconnect_weight_init=interconnect_weight_init,
                    interconnect_bias_init=interconnect_bias_init,
                    dtype=dtype,
                    key=k,
                )
            )
        self.layers = tuple(layers)

        self.x_size = x_size
        self.p_size = p_size
        self.out_size = out_size
        self.width_sizes = width_sizes
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias
        self.use_passthrough = use_passthrough
        self.non_decreasing = non_decreasing

    def __call__(
        self,
        x: Float[Array, "... x_size"],
        p: Float[Array, "... p_size"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "... out_size"]:
        """Forward pass through the `PICNN`.

        Args:
            x: A JAX array with shape `(x_size,)`. (Or shape `()` if
                `x_size="scalar"`.) The output will be element-wise
                convex in this input.
            p: A JAX array with shape `(p_size,)`. (Or shape `()` if
                `x_size="scalar"`.) The output can have an arbitrary
                relationship to this input.
            key: Ignored; provided for compatibility with the rest of the
                Equinox API.
                Defaults to None.

        Returns:
            A JAX array with shape `(out_size,)`. (Or shape `()` if
            `out_size="scalar"`.)

        """
        y = x
        u = p

        for layer in self.layers:
            y, u, _ = layer(y, u, x)

        return y
