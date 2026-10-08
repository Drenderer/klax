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

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Scalar

import klax


@pytest.fixture
def model():
    class _Model(eqx.Module):
        """Simple model: y = w * x + b."""

        weight: jnp.ndarray
        bias: jnp.ndarray

        def __init__(self, key=None):
            self.weight = jnp.array(1.0)
            self.bias = jnp.array(0.0)

        def __call__(self, x):
            return self.weight * x + self.bias

    return _Model()


# ===---------------------------------------------------------------------=== #
# klax.Loss
# ===---------------------------------------------------------------------=== #


class TestLossABC:
    def test_custom_subclass(self, model):
        """Test creating a custom Loss subclass."""

        class SquaredDifferenceLoss(klax.Loss):
            def value(self, model, batch, run_state):
                x, y = batch
                y_pred = jax.vmap(model)(x)
                sum = jnp.sum(jnp.square(y_pred - y))
                return sum, {"sum": sum}

        loss_fn = SquaredDifferenceLoss()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0, 3.0])
        batch = (x, y)

        result, aux = loss_fn(model, batch, 0)
        assert isinstance(result, jnp.ndarray)
        assert result.shape == ()
        assert "sum" in aux
        assert isinstance(aux["sum"], jnp.ndarray)
        assert aux["sum"].shape == ()

    def test_value_unwraps_model(self):
        """Test that value() unwraps constrained models."""

        class MyUnwrappable(klax.Unwrappable):
            array: jnp.ndarray

            def unwrap(self):
                return jnp.array(0.0)

        class TestModel(eqx.Module):
            param: MyUnwrappable

        class SimpleLoss(klax.Loss):
            def value(self, model, batch, run_state):
                param = model.param
                return param, {}

        loss_fn = SimpleLoss()
        model = TestModel(param=MyUnwrappable(array=jnp.array(1.0)))

        value, _ = loss_fn(model, None, 0)
        assert jnp.array_equal(value, jnp.array(0.0))

    def test_value_and_grad_returns_tuple(self, model):
        """Test that value_and_grad returns (value, grad) tuple."""

        class SimpleLoss(klax.Loss):
            def value(self, model, batch, run_state):
                x, y = batch
                y_pred = jax.vmap(model)(x)
                mse = jnp.mean(jnp.square(y_pred - y))
                return mse, {"mse": mse}

        loss_fn = SimpleLoss()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.5, 2.5, 3.5])
        batch = (x, y)

        (value, _), grad = loss_fn.value_and_grad(model, batch, 0)
        assert isinstance(value, jnp.ndarray)
        assert value.shape == ()
        # Grad should be a pytree matching model structure
        assert hasattr(grad, "weight")
        assert hasattr(grad, "bias")

    def test_value_and_grad_computes_correct_gradients(self, model):
        """Test that gradients are computed correctly."""

        class SimpleLoss(klax.Loss):
            def value(self, model, batch, run_state):
                x, y = batch
                y_pred = jax.vmap(model)(x)
                mse = jnp.mean(jnp.square(y_pred - y))
                return mse, {"mse": mse}

        loss_fn = SimpleLoss()
        # Data: y = 2*x + 1, so optimal weight ≈ 2, bias ≈ 1
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([3.0, 5.0, 7.0])
        batch = (x, y)

        (value, _), grad = loss_fn.value_and_grad(model, batch, 0)
        # Initial model: weight=1, bias=0, so preds are [1, 2, 3]
        # Residuals: [2, 3, 4], so loss = mean([4, 9, 16]) = 9.67
        assert value > 0
        # Gradients should be non-zero (model is not optimal)
        assert not jnp.allclose(grad.weight, 0.0)
        assert not jnp.allclose(grad.bias, 0.0)

    def test_partitioned_value(self, model):
        """Test partitioned_value method with split model."""

        class SimpleLoss(klax.Loss):
            def value(self, model, batch, run_state):
                x, y = batch
                y_pred = jax.vmap(model)(x)
                mse = jnp.mean(jnp.square(y_pred - y))
                return mse, {"mse": mse}

        loss_fn = SimpleLoss()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0, 3.0])
        batch = (x, y)

        # Partition the model
        params, static = eqx.partition(model, eqx.is_inexact_array)
        # Test that partitioned_value gives same result as value
        partitioned_result = loss_fn.partitioned_value(params, static, batch, 0)
        (full_result, _) = loss_fn(model, batch, 0)

        assert jnp.allclose(partitioned_result, full_result)


# ===---------------------------------------------------------------------=== #
# klax.loss
# ===---------------------------------------------------------------------=== #


class TestLossDecorator:
    def test_returns_loss_object(self):
        """Test that @loss decorator creates a Loss object."""

        @klax.loss
        def custom_loss(model, batch, run_state):
            x, y = batch
            y_pred = jax.vmap(model)(x)
            mse = jnp.mean(jnp.square(y_pred - y))
            return mse, {"mse": mse}

        assert isinstance(custom_loss, klax.Loss)

    def test_preserves_function_name(self):
        """Test that decorator preserves function metadata."""

        @klax.loss
        def my_custom_loss(model, batch, run_state):
            """Calculate my loss."""
            x, y = batch
            y_pred = jax.vmap(model)(x)
            mse = jnp.mean(jnp.square(y_pred - y))
            return mse, {"mse": mse}

        assert my_custom_loss.value.__doc__ == "Calculate my loss."

    def test_callable(self, model):
        """Test that decorated loss can be used for training."""

        @klax.loss
        def custom_loss(model, batch, run_state):
            x, y = batch
            y_pred = jax.vmap(model)(x)
            mse = jnp.mean(jnp.square(y_pred - y))
            return mse, {"mse": mse}

        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0, 3.0])
        batch = (x, y)

        # Test value method
        (value, _), _ = custom_loss.value_and_grad(model, batch, 0)
        assert isinstance(value, jnp.ndarray)

        # Test value_and_grad method
        (value, _), grad = custom_loss.value_and_grad(model, batch, 0)
        assert isinstance(value, jnp.ndarray)
        assert hasattr(grad, "weight")


# ===---------------------------------------------------------------------=== #
# klax.mse; klax.mae
# ===---------------------------------------------------------------------=== #


@dataclass
class IOCase:
    loss: klax.Loss
    zero_value: Scalar
    non_zero_value: Scalar


@pytest.fixture(
    scope="class",
    params=[
        pytest.param(
            IOCase(
                loss=klax.mse,
                zero_value=jnp.array(0.0),
                non_zero_value=jnp.array(4.0),
            ),
            id="mse",
        ),
        pytest.param(
            IOCase(
                loss=klax.mae,
                zero_value=jnp.array(0.0),
                non_zero_value=jnp.array(2.0),
            ),
            id="mae",
        ),
    ],
)
def impl(request):
    return request.param


class TestDefaultLosses:
    def test_zero_value(self, model, impl):
        """Test basic MSE computation."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0, 3.0])
        batch = (x, y)

        result, aux = impl.loss.value(model, batch, run_state=None)
        assert jnp.allclose(result, impl.zero_value)

    def test_non_zero_value(self, model, impl):
        """Test MSE with non-zero error."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([3.0, 4.0, 5.0])
        batch = (x, y)

        result, aux = impl.loss.value(model, batch, run_state=None)
        assert jnp.allclose(result, impl.non_zero_value)

    def test_value_and_grad(self, model, impl):
        """Test MSE gradient computation."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 3.0, 4.0])
        batch = (x, y)

        (value, aux), grad = impl.loss.value_and_grad(
            model, batch, run_state=None
        )
        assert isinstance(value, Array)
        assert hasattr(grad, "weight")
        assert hasattr(grad, "bias")
        # With uniform error, we expect non-zero gradients
        assert not jnp.allclose(grad.weight, 0.0)
        assert not jnp.allclose(grad.bias, 0.0)

    def test_batched(self, model, impl):
        """Test MSE with different batch axes specification."""
        # Batch on axis 0
        x = jnp.array([[1.0], [2.0], [3.0]])
        y = jnp.array([[1.0], [2.0], [3.0]])
        batch = (x, y)

        result, aux = impl.loss.value(model, batch, run_state=None)
        assert isinstance(result, Scalar)
        assert result.shape == ()

    def test_scalar_input(self, model, impl):
        """Test MSE with scalar vs vector inputs."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0, 3.0])
        batch = (x, y)

        result, aux = impl.loss.value(model, batch, run_state=None)
        assert jnp.allclose(result, 0.0)
