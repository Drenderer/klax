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
from klax.nn._icnn import PICNNLayer


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

    def test_enforce_nonnegative_true(self, getkey):
        """Test that enforce_nonnegative=True applies NonNegative wrapper."""
        layer = PICNNLayer(
            y_in_size=3,
            u_in_size=2,
            x_size=4,
            y_out_size=5,
            u_out_size=6,
            enforce_nonnegative=True,
            use_passthrough=True,
            key=getkey(),
        )

        # Check that the first weight in linear_y is wrapped with NonNegative
        assert isinstance(layer.linear_y.weights[0], klax.NonNegative)
        # The second and third weights (x and u paths) should not be wrapped
        assert isinstance(layer.linear_y.weights[1], jax.Array)
        assert isinstance(layer.linear_y.weights[2], jax.Array)

        # Verify the wrapped weight is actually non-negative
        finalized_layer = klax.finalize(layer)
        weight = finalized_layer.linear_y.weights[0]
        assert jnp.all(weight >= 0.0)

    def test_enforce_nonnegative_false(self, getkey):
        """Test that enforce_nonnegative=False does not apply wrapper."""
        layer = PICNNLayer(
            y_in_size=3,
            u_in_size=2,
            x_size=4,
            y_out_size=5,
            u_out_size=6,
            enforce_nonnegative=False,
            use_passthrough=True,
            key=getkey(),
        )

        # Check that none of the weights in linear_y are wrapped
        assert isinstance(layer.linear_y.weights[0], jax.Array)
        assert isinstance(layer.linear_y.weights[1], jax.Array)
        assert isinstance(layer.linear_y.weights[2], jax.Array)

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
