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

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from jax.nn.initializers import he_normal, uniform

import klax
from klax.nn import (
    FICNN,
    MLP,
    ConstantMatrix,
    ConstantSkewSymmetricMatrix,
    ConstantSPDMatrix,
    InputSplitLinear,
    Linear,
    Matrix,
    SkewSymmetricMatrix,
    SPDMatrix,
)
from klax.nn._icnn import PICNN, PICNNLayer


def test_linear(getkey, getzerowrap):
    # Zero input shape
    linear = Linear(0, 4, uniform(), key=getkey())
    x = jrandom.normal(getkey(), (0,))
    assert linear(x).shape == (4,)

    # Zero output shape
    linear = Linear(4, 0, uniform(), key=getkey())
    x = jrandom.normal(getkey(), (4,))
    assert linear(x).shape == (0,)

    # Positional arguments
    linear = Linear(3, 4, uniform(), key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    # Some keyword arguments
    linear = Linear(3, out_features=4, weight_init=uniform(), key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    # All keyword arguments
    linear = Linear(
        in_features=3, out_features=4, weight_init=uniform(), key=getkey()
    )
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    # Scalar shapes
    linear = Linear("scalar", 2, uniform(), key=getkey())
    x = jrandom.normal(getkey(), ())
    assert linear(x).shape == (2,)

    linear = Linear(2, "scalar", uniform(), key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert linear(x).shape == ()

    # Wrappers
    linear = Linear(
        3,
        4,
        uniform(),
        weight_wrap=getzerowrap,
        bias_wrap=getzerowrap,
        key=getkey(),
    )
    x = jrandom.normal(getkey(), (3,))
    assert jnp.all(klax.finalize(linear)(x) == 0.0)

    # Data type
    linear = Linear(2, "scalar", uniform(), key=getkey(), dtype=jnp.float16)
    x = jrandom.normal(getkey(), (2,), dtype=jnp.float16)
    assert linear(x).dtype == jnp.float16

    linear = Linear(
        2,
        "scalar",
        he_normal(),  # since uniform does not accept complex numbers
        key=getkey(),
        dtype=jnp.complex64,
    )
    x = jrandom.normal(getkey(), (2,), dtype=jnp.complex64)
    assert linear(x).dtype == jnp.complex64


def test_is_linear(getkey):
    # Zero input length
    is_linear = InputSplitLinear((0,), 4, uniform(), key=getkey())
    x = jrandom.normal(getkey(), (0,))
    assert is_linear(x).shape == (4,)

    is_linear = InputSplitLinear((0, 0), 4, uniform(), key=getkey())
    x = jrandom.normal(getkey(), (0,))
    assert is_linear(x, x).shape == (4,)

    # Zero length output
    is_linear = InputSplitLinear((2,), 0, uniform(), key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert is_linear(x).shape == (0,)

    # One non-zero input
    is_linear = InputSplitLinear((3,), 4, uniform(), key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert is_linear(x).shape == (4,)

    # Multiple non-zero inputs
    is_linear = InputSplitLinear((3, 2, 5), 4, uniform(), key=getkey())
    x0 = jrandom.normal(getkey(), (3,))
    x1 = jrandom.normal(getkey(), (2,))
    x2 = jrandom.normal(getkey(), (5,))
    assert is_linear(x0, x1, x2).shape == (4,)

    # Scalar shapes
    is_linear = InputSplitLinear(
        ("scalar", 2), 3, (uniform(), uniform()), key=getkey()
    )
    y = jrandom.normal(getkey(), ())
    z = jrandom.normal(getkey(), (2,))
    assert is_linear(y, z).shape == (3,)

    is_linear = InputSplitLinear((2, 3), "scalar", uniform(), key=getkey())
    y = jrandom.normal(getkey(), (2,))
    z = jrandom.normal(getkey(), (3,))
    assert is_linear(y, z).shape == ()

    # Weight wrappers
    is_linear = InputSplitLinear(
        (2, 3),
        "scalar",
        uniform(),
        weight_wraps=[klax.NonNegative, None],
        key=getkey(),
    )
    assert isinstance(is_linear.weights[0], klax.NonNegative)
    assert isinstance(is_linear.weights[1], jax.Array)

    # Data types
    for dtype in [jnp.float16, jnp.float32, jnp.complex64]:
        is_linear = InputSplitLinear(
            (2, 3), "scalar", he_normal(), key=getkey(), dtype=dtype
        )
        y = jrandom.normal(getkey(), (2,), dtype=dtype)
        z = jrandom.normal(getkey(), (3,), dtype=dtype)
        assert is_linear(y, z).dtype == dtype


def test_mlp(getkey):
    mlp = MLP(2, 3, 2 * [8], uniform(), key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)

    mlp = MLP(
        in_size=2,
        out_size=3,
        width_sizes=2 * [8],
        weight_init=uniform(),
        bias_init=uniform(),
        key=getkey(),
    )
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)

    mlp = MLP("scalar", 2, 2 * [2], uniform(), key=getkey())
    x = jrandom.normal(getkey(), ())
    assert mlp(x).shape == (2,)

    mlp = MLP(2, "scalar", 2 * [2], uniform(), key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == ()
    assert [mlp.layers[i].use_bias for i in range(0, 3)] == [True, True, True]

    mlp = MLP(
        2,
        3,
        2 * [8],
        uniform(),
        use_bias=False,
        use_final_bias=True,
        key=getkey(),
    )
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)
    assert [mlp.layers[i].use_bias for i in range(0, 3)] == [
        False,
        False,
        True,
    ]

    mlp = MLP(
        2,
        3,
        2 * [8],
        uniform(),
        use_bias=True,
        use_final_bias=False,
        key=getkey(),
    )
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)
    assert [mlp.layers[i].use_bias for i in range(0, 3)] == [True, True, False]

    mlp = MLP(
        2,
        3,
        [4, 8],
        uniform(),
        use_bias=True,
        use_final_bias=False,
        key=getkey(),
    )
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)
    assert [mlp.layers[i].in_features for i in range(0, 3)] == [2, 4, 8]
    assert [mlp.layers[i].out_features for i in range(0, 3)] == [4, 8, 3]


@pytest.mark.parametrize("use_passthrough", [True, False])
@pytest.mark.parametrize("non_decreasing", [True, False])
def test_ficnn(getkey, use_passthrough, non_decreasing):
    x = jrandom.normal(
        getkey(), (100, 2)
    )  # Sample 100 random evaluation points
    ficnn = klax.finalize(
        FICNN(
            2,
            "scalar",
            1 * [8],
            use_passthrough=use_passthrough,
            non_decreasing=non_decreasing,
            key=getkey(),
        )
    )

    # Assert expected output shape
    assert ficnn(x[0]).shape == ()
    # Assert the non-decreasing property
    if non_decreasing:
        grad_fun = jax.vmap(jax.grad(ficnn))
        assert jnp.all(grad_fun(x) >= 0)
    # Assert convexity: Check that the Hessian is positive definite but allow
    # for small numerical errors
    hessian_fun = jax.vmap(jax.hessian(ficnn))
    assert jnp.all(jnp.linalg.eigvals(hessian_fun(x)) > -1e-6)


def test_matrices(getkey):
    x = jrandom.normal(getkey(), (4,))

    m = Matrix(4, (1, 2, 3), [8], key=getkey())
    assert m(x).shape == (1, 2, 3)
    m = Matrix(in_size="scalar", shape=(5, 3), width_sizes=[8], key=getkey())
    assert m(jnp.array(0.0)).shape == (5, 3)

    m = ConstantMatrix(4, key=getkey())
    assert m(x).shape == (4, 4)
    m = ConstantMatrix(shape=(1, 2, 3), key=getkey())
    assert m(x).shape == (1, 2, 3)

    m = SkewSymmetricMatrix(4, (2, 3, 3), [8], key=getkey())
    output = m(x)
    assert output.shape == (2, 3, 3)
    assert jnp.allclose(output, -jnp.matrix_transpose(output))
    m = SkewSymmetricMatrix(
        in_size="scalar", shape=(5, 3, 3), width_sizes=[8], key=getkey()
    )
    assert klax.finalize(m)(0.0).shape == (5, 3, 3)
    assert jnp.allclose(output, -jnp.matrix_transpose(output))

    m = ConstantSkewSymmetricMatrix(4, key=getkey())
    output = klax.finalize(m)(x)
    assert output.shape == (4, 4)
    assert jnp.allclose(output, -jnp.matrix_transpose(output))
    m = ConstantSkewSymmetricMatrix((2, 3, 3), key=getkey())
    output = klax.finalize(m)(x)
    assert output.shape == (2, 3, 3)
    assert jnp.allclose(output, -jnp.matrix_transpose(output))

    m = SPDMatrix(4, (2, 3, 3), [8], dtype=jnp.complex64, key=getkey())
    output = m(x)
    assert output.shape == (2, 3, 3)
    assert jnp.allclose(output, jnp.conjugate(output.mT))
    assert jnp.all(jnp.linalg.eigvalsh(output) > 0.0)
    m = SPDMatrix(
        in_size="scalar", shape=(5, 3, 3), width_sizes=[8], key=getkey()
    )
    assert m(jnp.array(0.0)).shape == (5, 3, 3)
    assert jnp.allclose(output, jnp.conjugate(output.mT))

    m = ConstantSPDMatrix(4, key=getkey())
    output = m(x)
    assert output.shape == (4, 4)
    assert jnp.allclose(output, jnp.conjugate(output.mT))
    assert jnp.all(jnp.linalg.eigvalsh(output) > 0.0)
    m = ConstantSPDMatrix((2, 3, 3), dtype=jnp.complex64, key=getkey())
    output = m(x)
    assert output.shape == (2, 3, 3)
    assert jnp.allclose(output, jnp.conjugate(output.mT))
    assert jnp.all(jnp.linalg.eigvalsh(output) > 0.0)


class TestPICNNLayer:
    """Test suite for the PICNNLayer implementation."""

    def test_basic_shapes(self, getkey):
        """Test that the layer outputs the correct shapes."""
        layer = PICNNLayer(
            y_in_size=3,
            u_in_size=2,
            x_size=4,
            y_out_size=5,
            u_out_size=6,
            key=getkey(),
        )
        layer = klax.finalize(layer)

        y = jrandom.normal(getkey(), (3,))
        u = jrandom.normal(getkey(), (2,))
        x = jrandom.normal(getkey(), (4,))

        y_out, u_out, x_out = layer(y, u, x)
        assert y_out.shape == (5,)
        assert u_out.shape == (6,)
        assert x_out.shape == (4,)

    @pytest.mark.parametrize(
        "nonnegative_y_weight, expected_type",
        [
            pytest.param(False, jax.Array, id="no_wrapper"),
            pytest.param(True, klax.NonNegative, id="with_wrapper"),
        ],
    )
    def test_nonnegative_y_weight(
        self, nonnegative_y_weight, expected_type, getkey
    ):
        layer = PICNNLayer(
            y_in_size=3,
            u_in_size=2,
            x_size=4,
            y_out_size=5,
            u_out_size=6,
            nonnegative_y_weight=nonnegative_y_weight,
            use_passthrough=True,
            key=getkey(),
        )

        assert isinstance(layer.linear_y.weights[0], expected_type)

    def test_use_passthrough_true(self, getkey):
        """Test that use_passthrough=True creates linear_xu layer."""
        layer = PICNNLayer(
            y_in_size=3,
            u_in_size=2,
            x_size=4,
            y_out_size=5,
            u_out_size=6,
            use_passthrough=True,
            key=getkey(),
        )

        # linear_xu should exist
        assert layer.linear_xu is not None
        assert isinstance(layer.linear_xu, Linear)

        # linear_y should be InputSplitLinear with 3 inputs
        assert isinstance(layer.linear_y, InputSplitLinear)
        assert len(layer.linear_y.weights) == 3

    def test_use_passthrough_false(self, getkey):
        """Test that use_passthrough=False does not create linear_xu."""
        layer = PICNNLayer(
            y_in_size=3,
            u_in_size=2,
            x_size=4,
            y_out_size=5,
            u_out_size=6,
            use_passthrough=False,
            key=getkey(),
        )

        # linear_xu should not exist
        assert layer.linear_xu is None

        # linear_y should be InputSplitLinear with only 2 inputs
        assert isinstance(layer.linear_y, InputSplitLinear)
        assert len(layer.linear_y.weights) == 2

    def test_use_bias_true(self, getkey):
        """Test that use_bias=True adds bias to linear_y."""
        layer = PICNNLayer(
            y_in_size=3,
            u_in_size=2,
            x_size=4,
            y_out_size=5,
            u_out_size=6,
            use_bias=True,
            key=getkey(),
        )

        # linear_y should have a bias
        assert layer.linear_y.use_bias is True
        assert layer.linear_y.bias is not None

    def test_use_bias_false(self, getkey):
        """Test that use_bias=False removes bias from linear_y."""
        layer = PICNNLayer(
            y_in_size=3,
            u_in_size=2,
            x_size=4,
            y_out_size=5,
            u_out_size=6,
            use_bias=False,
            key=getkey(),
        )

        # linear_y should not have a bias
        assert layer.linear_y.use_bias is False
        assert layer.linear_y.bias is None

    def test_batched_call_single_batch(self, getkey, allow_rank_promotion):
        """Test that the layer supports batched calls (single batch dim)."""
        layer = klax.finalize(
            PICNNLayer(
                y_in_size=3,
                u_in_size=2,
                x_size=4,
                y_out_size=5,
                u_out_size=6,
                key=getkey(),
            )
        )

        batch_size = 10
        y = jrandom.normal(getkey(), (batch_size, 3))
        u = jrandom.normal(getkey(), (batch_size, 2))
        x = jrandom.normal(getkey(), (batch_size, 4))

        y_out, u_out, x_out = layer(y, u, x)
        assert y_out.shape == (batch_size, 5)
        assert u_out.shape == (batch_size, 6)
        assert x_out.shape == (batch_size, 4)

    def test_batched_call_multiple_batch_dims(
        self, getkey, allow_rank_promotion
    ):
        """Test that the layer supports multiple batch dimensions."""
        layer = klax.finalize(
            PICNNLayer(
                y_in_size=3,
                u_in_size=2,
                x_size=4,
                y_out_size=5,
                u_out_size=6,
                key=getkey(),
            )
        )

        batch_shape = (7, 10)
        y = jrandom.normal(getkey(), batch_shape + (3,))
        u = jrandom.normal(getkey(), batch_shape + (2,))
        x = jrandom.normal(getkey(), batch_shape + (4,))

        y_out, u_out, x_out = layer(y, u, x)
        assert y_out.shape == batch_shape + (5,)
        assert u_out.shape == batch_shape + (6,)
        assert x_out.shape == batch_shape + (4,)

    def test_custom_activations(self, getkey):
        """Test that custom activation functions work correctly."""
        layer = klax.finalize(
            PICNNLayer(
                y_in_size=3,
                u_in_size=2,
                x_size=4,
                y_out_size=5,
                u_out_size=6,
                activation_y=jax.nn.relu,
                activation_u=jax.nn.tanh,
                activation_yu=jax.nn.sigmoid,
                activation_xu=jnp.square,
                key=getkey(),
            )
        )

        y = jrandom.normal(getkey(), (3,))
        u = jrandom.normal(getkey(), (2,))
        x = jrandom.normal(getkey(), (4,))

        # Just verify it runs without error
        y_out, u_out, x_out = layer(y, u, x)
        assert y_out.shape == (5,)
        assert u_out.shape == (6,)
        assert x_out.shape == (4,)

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
    def test_dtype_preservation(self, dtype, getkey):
        """Test that the layer preserves the specified dtype."""
        layer = klax.finalize(
            PICNNLayer(
                y_in_size=3,
                u_in_size=2,
                x_size=4,
                y_out_size=5,
                u_out_size=6,
                dtype=dtype,
                key=getkey(),
            )
        )

        y = jrandom.normal(getkey(), (3,), dtype=dtype)
        u = jrandom.normal(getkey(), (2,), dtype=dtype)
        x = jrandom.normal(getkey(), (4,), dtype=dtype)

        y_out, u_out, x_out = layer(y, u, x)
        assert y_out.dtype == dtype
        assert u_out.dtype == dtype
        assert x_out.dtype == dtype

    def test_x_passthrough_unchanged(self, getkey):
        """Test that x is passed through unchanged."""
        layer = klax.finalize(
            PICNNLayer(
                y_in_size=3,
                u_in_size=2,
                x_size=4,
                y_out_size=5,
                u_out_size=6,
                key=getkey(),
            )
        )

        y = jrandom.normal(getkey(), (3,))
        u = jrandom.normal(getkey(), (2,))
        x = jrandom.normal(getkey(), (4,))

        y_out, u_out, x_out = layer(y, u, x)
        # x should be returned unchanged
        assert jnp.allclose(x_out, x)

    def test_interconnect_initialization(self, getkey):
        """Test custom interconnect weight and bias initialization."""
        layer = PICNNLayer(
            y_in_size=3,
            u_in_size=2,
            x_size=4,
            y_out_size=5,
            u_out_size=6,
            interconnect_weight_init=uniform(),
            interconnect_bias_init=uniform(),
            key=getkey(),
        )

        # Just verify the layer is created without error
        assert layer.linear_yu is not None
        if layer.use_passthrough:
            assert layer.linear_xu is not None

    def test_trainable_activation_functions(
        self, getkey, allow_rank_promotion
    ):
        """Test that each neuron's activation receives independent parameter updates.

        This test verifies that when using learnable activation functions with vmapped
        parameters, each neuron gets its own copy of the parameters that receive
        independent gradient updates.
        """

        # Define a learnable activation function with scale and bias parameters
        class LearnableActivation(eqx.Module):
            bias: jax.Array

            def __init__(self, initial_bias=0.0, *, key):
                self.bias = jnp.array(initial_bias)

            def __call__(self, x):
                return jax.nn.softplus(x) + self.bias

        # Create learnable activations
        activation = LearnableActivation(initial_bias=0.0, key=getkey())

        # Create a PICNN layer with learnable activation
        layer = PICNNLayer(
            y_in_size=3,
            u_in_size=2,
            x_size=4,
            y_out_size=5,  # 5 neurons, each with independent activation params
            u_out_size=6,
            activation_y=activation,
            activation_u=activation,
            activation_yu=activation,
            activation_xu=activation,
            key=getkey(),
        )

        # Create a simple loss function that is sensitive to individual neuron outputs
        def loss_fn(layer, y, u, x):
            layer = klax.unwrap(layer)
            y_out, u_out, x_out = layer(y, u, x)
            targets = jnp.array([1.0, -1.0, 0.5, -0.5, 2.0])
            return jnp.mean((y_out - targets) ** 2)

        # Prepare inputs
        y = jrandom.normal(getkey(), (3,))
        u = jrandom.normal(getkey(), (2,))
        x = jrandom.normal(getkey(), (4,))

        # Compute gradients with respect to the layer parameters
        grads = eqx.filter_grad(loss_fn)(layer, y, u, x)

        # Verify that the gradient for each bias parameter in the loss functions
        # is per neuron
        assert grads.activation_y.bias.shape == (5,)
        assert grads.activation_u.bias.shape == (6,)
        assert grads.activation_yu.bias.shape == (3,)
        assert grads.activation_xu.bias.shape == (4,)


class TestPICNN:
    """Test suite for the PICNN implementation.

    Tests focus on features specific to PICNN that are not covered by
    PICNNLayer tests, such as stacking multiple layers and composition.
    """

    def test_scalar_sizes(self, getkey):
        """Test PICNN with scalar input/output sizes."""
        picnn = klax.finalize(
            PICNN(
                x_size="scalar",
                p_size="scalar",
                out_size="scalar",
                width_sizes=[(4, 5)],
                key=getkey(),
            )
        )

        x = jnp.array(1.5)
        p = jnp.array(0.5)

        y = picnn(x, p)
        assert y.shape == ()

    @pytest.mark.parametrize(
        "width_sizes",
        [
            [2, 3, 4],
            [(2, 3)],
            [(4, 5), (6, 7)],
            [(3, 4), (5, 6), (7, 8), (9, 10)],
        ],
    )
    def test_different_width_sizes(self, width_sizes, getkey):
        """Test PICNN with various width size configurations."""
        picnn = klax.finalize(
            PICNN(
                x_size=3,
                p_size=2,
                out_size=1,
                width_sizes=width_sizes,
                key=getkey(),
            )
        )

        x = jrandom.normal(getkey(), (3,))
        p = jrandom.normal(getkey(), (2,))

        y = picnn(x, p)
        assert y.shape == (1,)
        assert jnp.isfinite(y).all()

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
    def test_dtype_preservation(self, dtype, getkey):
        """Test that PICNN preserves specified dtype across layers."""
        picnn = klax.finalize(
            PICNN(
                x_size=3,
                p_size=2,
                out_size=2,
                width_sizes=[(4, 5)],
                dtype=dtype,
                key=getkey(),
            )
        )

        x = jrandom.normal(getkey(), (3,), dtype=dtype)
        p = jrandom.normal(getkey(), (2,), dtype=dtype)

        y = picnn(x, p)
        assert y.dtype == dtype

    def test_batched_inputs(self, getkey, allow_rank_promotion):
        """Test that PICNN handles batched inputs correctly."""
        picnn = klax.finalize(
            PICNN(
                x_size=3,
                p_size=2,
                out_size=2,
                width_sizes=[(4, 5), (6, 7)],
                key=getkey(),
            )
        )

        batch_size = 8
        x = jrandom.normal(getkey(), (batch_size, 3))
        p = jrandom.normal(getkey(), (batch_size, 2))

        y = picnn(x, p)
        assert y.shape == (batch_size, 2)

    @pytest.mark.parametrize("use_passthrough", [True, False])
    def test_use_passthrough(self, use_passthrough, getkey):
        """Test PICNN with and without use_passthrough."""
        picnn = PICNN(
            x_size=3,
            p_size=2,
            out_size=2,
            width_sizes=[(4, 5)],
            use_passthrough=use_passthrough,
            key=getkey(),
        )

        assert picnn.use_passthrough is use_passthrough
        for layer in picnn.layers:
            assert layer.use_passthrough is use_passthrough

    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize("use_final_bias", [True, False])
    def test_use_bias_configurations(self, use_bias, use_final_bias, getkey):
        """Test PICNN with different bias configurations."""
        picnn = PICNN(
            x_size=3,
            p_size=2,
            out_size=2,
            width_sizes=[(4, 5)],
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=getkey(),
        )
        assert picnn.use_bias is use_bias
        assert picnn.use_final_bias is use_final_bias
        for layer in picnn.layers[:-1]:
            assert layer.use_bias is use_bias
        assert picnn.layers[-1].use_bias is use_final_bias

    @pytest.mark.parametrize("non_decreasing", [True, False])
    def test_non_decreasing_property(self, non_decreasing, getkey):
        """Test PICNN with non_decreasing=True."""
        picnn = PICNN(
            x_size=2,
            p_size=2,
            out_size=1,
            width_sizes=[(4, 5)],
            non_decreasing=non_decreasing,
            key=getkey(),
        )

        assert picnn.non_decreasing is non_decreasing
        # The first layer should enforce non-negative constraint
        assert picnn.layers[0].nonnegative_y_weight is non_decreasing
        for layer in picnn.layers[1:]:
            assert layer.nonnegative_y_weight is True
        for layer in picnn.layers:
            assert layer.nonnegative_passthrough is non_decreasing

    def test_custom_activations(self, getkey):
        """Test PICNN with custom activation functions."""
        picnn = klax.finalize(
            PICNN(
                x_size=3,
                p_size=2,
                out_size=2,
                width_sizes=[(4, 5), (6, 7)],
                activation_y=jax.nn.relu,
                activation_u=jax.nn.tanh,
                activation_yu=jax.nn.sigmoid,
                activation_xu=jnp.square,
                final_activation_y=jax.nn.softplus,
                key=getkey(),
            )
        )

        x = jrandom.normal(getkey(), (3,))
        p = jrandom.normal(getkey(), (2,))

        y = picnn(x, p)
        assert y.shape == (2,)
        assert jnp.isfinite(y).all()

    def test_layer_count(self, getkey):
        """Test that PICNN creates correct number of layers."""
        width_sizes = [(4, 5), (6, 7), (8, 9)]
        picnn = PICNN(
            x_size=3,
            p_size=2,
            out_size=2,
            width_sizes=width_sizes,
            key=getkey(),
        )

        assert len(picnn.layers) == len(width_sizes) + 1

    @pytest.mark.parametrize("use_passthrough", [True, False])
    @pytest.mark.parametrize("non_decreasing", [True, False])
    def test_convexity(self, use_passthrough, non_decreasing, getkey):
        x = jrandom.normal(getkey(), (10, 5))
        p = jrandom.normal(getkey(), (10, 2))
        picnn = klax.finalize(
            PICNN(
                x_size=5,
                p_size=2,
                out_size="scalar",
                width_sizes=[(4, 5), (6, 7)],
                use_passthrough=use_passthrough,
                non_decreasing=non_decreasing,
                interconnect_weight_init=jax.nn.initializers.he_normal(),
                activation_xu=jax.nn.softplus,
                key=getkey(),
            )
        )

        # Assert the non-decreasing property
        if non_decreasing:
            grad_fun = jax.vmap(jax.grad(picnn))
            assert jnp.all(grad_fun(x, p) >= 0)

        # Assert convexity: Check that the Hessian is positive definite but allow
        # for small numerical errors
        hessian_fun = jax.vmap(jax.hessian(picnn))
        assert jnp.all(jnp.linalg.eigvals(hessian_fun(x, p)) > -1e-6)

    @pytest.mark.parametrize("use_passthrough", [True, False])
    @pytest.mark.parametrize("non_decreasing", [True, False])
    def test_initial_gradients(self, use_passthrough, non_decreasing, getkey):
        x = jrandom.normal(getkey(), (10, 5))
        p = jrandom.normal(getkey(), (10, 2))

        picnn = klax.finalize(
            PICNN(
                x_size=5,
                p_size=2,
                out_size="scalar",
                width_sizes=[(4, 5), (6, 7)],
                use_passthrough=use_passthrough,
                non_decreasing=non_decreasing,
                interconnect_weight_init=jax.nn.initializers.zeros,
                interconnect_bias_init=jax.nn.initializers.ones,
                key=getkey(),
            )
        )

        # Assert that the intial gradients with respect to p are zero
        grad_p_fun = jax.vmap(jax.grad(picnn, argnums=1))
        grad_p = grad_p_fun(x, p)
        assert jnp.all(grad_p == 0.0)

        # Assert that the intial gradients with respect to parameters
        # in the interconnection path are nonzero
        def loss_fn(picnn, x, p):
            picnn = klax.unwrap(picnn)
            y = jax.vmap(picnn)(x, p)
            targets = jnp.array(1.0)
            return jnp.mean((y - targets) ** 2)

        # Compute gradients with respect to the layer parameters
        grads = eqx.filter_grad(loss_fn)(picnn, x, p)

        for layer in grads.layers:
            assert not jnp.all(layer.linear_yu.weight == 0)
            assert not jnp.all(layer.linear_yu.bias == 0)
            if use_passthrough:
                assert not jnp.all(layer.linear_xu.weight == 0)
                assert not jnp.all(layer.linear_xu.bias == 0)
