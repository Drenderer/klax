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
import pytest

from klax._losses import Loss, loss, mae, mse
from klax._wrappers import Constraint, Unwrappable


class SimpleLinearModel(eqx.Module):
    """Simple model: y = w*x + b."""

    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, key=None):
        self.weight = jnp.array(1.0)
        self.bias = jnp.array(0.0)

    def __call__(self, x):
        return self.weight * x + self.bias


class TestLossAbstractClass:
    """Test the abstract Loss class and its methods."""

    def test_custom_loss_subclass(self):
        """Test creating a custom Loss subclass."""

        class SquaredDifferenceLoss(Loss):
            def __call__(self, model, batch, batch_axes):
                x, y = batch
                y_pred = jax.vmap(model, in_axes=batch_axes)(x)
                return jnp.sum(jnp.square(y_pred - y))

        loss_fn = SquaredDifferenceLoss()
        model = SimpleLinearModel()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0, 3.0])
        batch = (x, y)

        result = loss_fn.value(model, batch, 0)
        assert isinstance(result, jnp.ndarray)
        assert result.shape == ()

    def test_value_unwraps_model(self):
        """Test that value() unwraps constrained models."""

        class MinValue(Constraint):
            array: jnp.ndarray
            minval: jnp.ndarray

            def unwrap(self):
                return self.array

            def apply(self):
                return eqx.tree_at(
                    lambda x: x.array,
                    self,
                    replace=jnp.maximum(self.array, self.minval),
                )

        class TestModel(eqx.Module):
            param: Unwrappable

            def __init__(self):
                self.param = MinValue(
                    array=jnp.array(5.0), minval=jnp.array(0.0)
                )

            def __call__(self, x):
                return self.param * x

        class SimpleLoss(Loss):
            def __call__(self, model, batch, batch_axes):
                x, y = batch
                y_pred = jax.vmap(model, in_axes=batch_axes)(x)
                return jnp.mean(jnp.square(y_pred - y))

        loss_fn = SimpleLoss()
        model = TestModel()
        x = jnp.array([1.0, 2.0])
        y = jnp.array([5.0, 10.0])
        batch = (x, y)

        value = loss_fn.value(model, batch, 0)
        assert isinstance(value, jnp.ndarray)

    def test_value_and_grad_returns_tuple(self):
        """Test that value_and_grad returns (value, grad) tuple."""

        class SimpleLoss(Loss):
            def __call__(self, model, batch, batch_axes):
                x, y = batch
                y_pred = jax.vmap(model, in_axes=batch_axes)(x)
                return jnp.mean(jnp.square(y_pred - y))

        loss_fn = SimpleLoss()
        model = SimpleLinearModel()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.5, 2.5, 3.5])
        batch = (x, y)

        value, grad = loss_fn.value_and_grad(model, batch, 0)
        assert isinstance(value, jnp.ndarray)
        assert value.shape == ()
        # Grad should be a pytree matching model structure
        assert hasattr(grad, "weight")
        assert hasattr(grad, "bias")

    def test_value_and_grad_computes_correct_gradients(self):
        """Test that gradients are computed correctly."""

        class SimpleLoss(Loss):
            def __call__(self, model, batch, batch_axes):
                x, y = batch
                y_pred = jax.vmap(model, in_axes=batch_axes)(x)
                return jnp.mean(jnp.square(y_pred - y))

        loss_fn = SimpleLoss()
        model = SimpleLinearModel()
        # Data: y = 2*x + 1, so optimal weight ≈ 2, bias ≈ 1
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([3.0, 5.0, 7.0])
        batch = (x, y)

        value, grad = loss_fn.value_and_grad(model, batch, 0)
        # Initial model: weight=1, bias=0, so preds are [1, 2, 3]
        # Residuals: [2, 3, 4], so loss = mean([4, 9, 16]) = 9.67
        assert value > 0
        # Gradients should be non-zero (model is not optimal)
        assert not jnp.allclose(grad.weight, 0.0)
        assert not jnp.allclose(grad.bias, 0.0)

    def test_partitioned_value(self):
        """Test partitioned_value method with split model."""

        class SimpleLoss(Loss):
            def __call__(self, model, batch, batch_axes):
                x, y = batch
                y_pred = jax.vmap(model, in_axes=batch_axes)(x)
                return jnp.mean(jnp.square(y_pred - y))

        loss_fn = SimpleLoss()
        model = SimpleLinearModel()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0, 3.0])
        batch = (x, y)

        # Partition the model
        params, static = eqx.partition(model, eqx.is_inexact_array)
        # Test that partitioned_value gives same result as value
        partitioned_result = loss_fn.partitioned_value(
            params, static, batch, 0
        )
        full_result = loss_fn.value(model, batch, 0)

        assert jnp.allclose(partitioned_result, full_result)


class TestLossDecorator:
    """Test the @loss decorator."""

    def test_loss_decorator_creates_loss_object(self):
        """Test that @loss decorator creates a Loss object."""

        @loss
        def custom_loss(model, batch, batch_axes):
            x, y = batch
            y_pred = jax.vmap(model, in_axes=batch_axes)(x)
            return jnp.mean(jnp.square(y_pred - y))

        assert isinstance(custom_loss, Loss)

    def test_loss_decorator_preserves_function_name(self):
        """Test that decorator preserves function metadata."""

        @loss
        def my_custom_loss(model, batch, batch_axes):
            """Calculate my loss."""
            x, y = batch
            y_pred = jax.vmap(model, in_axes=batch_axes)(x)
            return jnp.mean(jnp.square(y_pred - y))

        assert my_custom_loss.__doc__ == "Calculate my loss."

    def test_loss_decorator_can_be_called(self):
        """Test that decorated loss can be used for training."""

        @loss
        def custom_loss(model, batch, batch_axes):
            x, y = batch
            y_pred = jax.vmap(model, in_axes=batch_axes)(x)
            return jnp.mean(jnp.square(y_pred - y))

        model = SimpleLinearModel()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0, 3.0])
        batch = (x, y)

        # Test value method
        value = custom_loss.value(model, batch, 0)
        assert isinstance(value, jnp.ndarray)

        # Test value_and_grad method
        value, grad = custom_loss.value_and_grad(model, batch, 0)
        assert isinstance(value, jnp.ndarray)
        assert hasattr(grad, "weight")


class TestMSELoss:
    """Test the mean squared error loss."""

    def test_mse_basic(self):
        """Test basic MSE computation."""
        model = SimpleLinearModel()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0, 3.0])
        batch = (x, y)

        result = mse.value(model, batch, 0)
        # Model predicts [1, 2, 3], y = [1, 2, 3], so loss = 0
        assert jnp.allclose(result, 0.0)

    def test_mse_non_zero_error(self):
        """Test MSE with non-zero error."""
        model = SimpleLinearModel()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 3.0, 4.0])
        batch = (x, y)

        result = mse.value(model, batch, 0)
        # Model predicts [1, 2, 3], y = [2, 3, 4]
        # Residuals: [1, 1, 1], squared = [1, 1, 1], mean = 1
        assert jnp.allclose(result, 1.0)

    def test_mse_with_gradient(self):
        """Test MSE gradient computation."""
        model = SimpleLinearModel()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 3.0, 4.0])
        batch = (x, y)

        value, grad = mse.value_and_grad(model, batch, 0)
        assert isinstance(value, jnp.ndarray)
        assert hasattr(grad, "weight")
        assert hasattr(grad, "bias")
        # With uniform error, we expect non-zero gradients
        assert not jnp.allclose(grad.weight, 0.0)
        assert not jnp.allclose(grad.bias, 0.0)

    def test_mse_with_batched_axes(self):
        """Test MSE with different batch axes specification."""
        model = SimpleLinearModel()
        # Batch on axis 0
        x = jnp.array([[1.0], [2.0], [3.0]])
        y = jnp.array([[1.0], [2.0], [3.0]])
        batch = (x, y)

        result = mse.value(model, batch, 0)
        assert isinstance(result, jnp.ndarray)
        assert result.shape == ()

    def test_mse_with_scalar_inputs(self):
        """Test MSE with scalar vs vector inputs."""
        model = SimpleLinearModel()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0, 3.0])
        batch = (x, y)

        # Using batch axis 0 (default for 1D arrays)
        result = mse.value(model, batch, 0)
        assert jnp.allclose(result, 0.0)


class TestMAELoss:
    """Test the mean absolute error loss."""

    def test_mae_basic(self):
        """Test basic MAE computation."""
        model = SimpleLinearModel()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0, 3.0])
        batch = (x, y)

        result = mae.value(model, batch, 0)
        # Model predicts [1, 2, 3], y = [1, 2, 3], so loss = 0
        assert jnp.allclose(result, 0.0)

    def test_mae_non_zero_error(self):
        """Test MAE with non-zero error."""
        model = SimpleLinearModel()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 3.0, 4.0])
        batch = (x, y)

        result = mae.value(model, batch, 0)
        # Model predicts [1, 2, 3], y = [2, 3, 4]
        # Residuals: [1, 1, 1], abs = [1, 1, 1], mean = 1
        assert jnp.allclose(result, 1.0)

    def test_mae_vs_mse_on_uniform_error(self):
        """Test that MAE and MSE differ with uniform errors."""
        model = SimpleLinearModel()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([3.0, 4.0, 5.0])
        batch = (x, y)

        mae_val = mae.value(model, batch, 0)
        mse_val = mse.value(model, batch, 0)
        # With uniform error of 2, MAE = 2, MSE = 4
        assert jnp.allclose(mae_val, 2.0)
        assert jnp.allclose(mse_val, 4.0)
        assert mae_val < mse_val

    def test_mae_with_gradient(self):
        """Test MAE gradient computation."""
        model = SimpleLinearModel()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 3.0, 4.0])
        batch = (x, y)

        value, grad = mae.value_and_grad(model, batch, 0)
        assert isinstance(value, jnp.ndarray)
        assert hasattr(grad, "weight")
        assert hasattr(grad, "bias")
        # With uniform error, we expect non-zero gradients
        assert not jnp.allclose(grad.weight, 0.0)
        assert not jnp.allclose(grad.bias, 0.0)

    def test_mae_with_batched_axes(self):
        """Test MAE with different batch axes specification."""
        model = SimpleLinearModel()
        x = jnp.array([[1.0], [2.0], [3.0]])
        y = jnp.array([[1.0], [2.0], [3.0]])
        batch = (x, y)

        result = mae.value(model, batch, 0)
        assert isinstance(result, jnp.ndarray)
        assert result.shape == ()


class TestLossComparison:
    """Test comparing different loss functions."""

    def test_mse_mae_agreement_on_zero_error(self):
        """Test that MSE and MAE agree when error is zero."""
        model = SimpleLinearModel()
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0, 3.0])
        batch = (x, y)

        mse_val = mse.value(model, batch, 0)
        mae_val = mae.value(model, batch, 0)
        assert jnp.allclose(mse_val, mae_val)
        assert jnp.allclose(mse_val, 0.0)

    def test_mse_mae_with_outliers(self):
        """Test MSE and MAE responses to outliers."""
        model = SimpleLinearModel()
        # Mostly small errors, one large error
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        y = jnp.array([1.1, 2.1, 3.1, 13.0])  # Last has error 9
        batch = (x, y)

        mse_val = mse.value(model, batch, 0)
        mae_val = mae.value(model, batch, 0)
        # MSE should be more affected by large error
        assert mse_val > mae_val
