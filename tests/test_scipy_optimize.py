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
import jax.random as jr
import numpy as np
import pytest
from jax.nn.initializers import uniform

import klax
from klax._scipy_optimize import (
    ScipyCallbackAdapter,
    ScipyModelAdapter,
    ScipyTrainingContext,
    ScipyTrainingState,
    scipy_fit,
    scipy_loss_wrapper,
)


class TestScipyModelAdapter:
    """Test the ScipyModelAdapter for model flattening and unflattening."""

    def test_adapter_roundtrip(self, getkey):
        """Test flattening and unflattening with an MLP."""
        model = klax.nn.MLP(3, 1, [8, 8], key=getkey())
        adapter = ScipyModelAdapter(model)

        # Flatten
        flat = adapter.flatten(model)
        assert isinstance(flat, jnp.ndarray)
        assert flat.ndim == 1

        # Unflatten
        unflatten_model = adapter.unflatten(flat)
        assert eqx.tree_equal(model, unflatten_model)

        # Flatten again
        flat2 = adapter.flatten(unflatten_model)
        assert np.array_equal(flat, flat2)

    def test_adapter_bounds(self, getkey):
        """Test bounds with NonNegative constraints."""

        class ModelWithConstraint(eqx.Module):
            param: klax.NonNegative
            regular: jnp.ndarray

        param = klax.NonNegative(jr.normal(getkey(), (3,)))
        regular = jr.normal(getkey(), (2,))
        model = ModelWithConstraint(param, regular)

        adapter = ScipyModelAdapter(model)

        # Should have bounds for both parameters and regular
        bounds = adapter.bounds
        assert len(bounds) == 5  # 3 from param + 2 from regular

        # First 3 bounds should be (0, inf)
        for lower, upper in bounds[:3]:
            assert lower == 0
            assert upper == np.inf

        # Last 2 bounds should be (-inf, inf)
        for lower, upper in bounds[3:]:
            assert lower == -np.inf
            assert upper == np.inf

    def test_adapter_non_trainable(self, getkey):
        """Test that static (non-trainable) parts are preserved."""

        class ModelWithNonTrainable(eqx.Module):
            param: jnp.ndarray
            static_param: klax.NonTrainable

        param = jr.normal(getkey(), (3,))
        static = klax.non_trainable(jr.normal(getkey(), (2,)))
        model = ModelWithNonTrainable(param, static)

        adapter = ScipyModelAdapter(model)

        # Flatten and unflatten
        flat = adapter.flatten(model)
        assert flat.size == 3

        reconstructed = adapter.unflatten(flat)
        assert eqx.tree_equal(model, reconstructed)


class TestScipyTrainingState:
    """Test the ScipyTrainingState class."""

    def test_state_model_property(self, getkey):
        """Test that model property correctly reconstructs the model."""
        model = klax.nn.Linear(2, 2, uniform(), key=getkey())
        adapter = ScipyModelAdapter(model)
        x = adapter.flatten(model)

        state = ScipyTrainingState(adapter, run_state=None, x=x)
        reconstructed = state.model

        assert eqx.tree_equal(model, reconstructed)

    def test_state_step_increment(self, getkey):
        """Test that step counter increments."""
        model = klax.nn.Linear(2, 2, uniform(), key=getkey())
        adapter = ScipyModelAdapter(model)
        x = adapter.flatten(model)

        state = ScipyTrainingState(adapter, run_state=None, x=x)
        assert state.step == 0

        # Simulate update with a modified x
        new_x = x + 0.1
        from scipy.optimize import OptimizeResult

        result = OptimizeResult(x=new_x, fun=1.0)
        state.update(result)

        assert state.step == 1


class TestScipyTrainingContext:
    """Test the ScipyTrainingContext class."""

    def test_context_batch_generator_raises(self, getkey):
        """Test that batch_generator raises NotImplementedError."""
        model = klax.nn.Linear(2, 2, uniform(), key=getkey())
        adapter = ScipyModelAdapter(model)
        x = adapter.flatten(model)

        context = ScipyTrainingContext(
            adapter, None, "SLSQP", 100, klax.mse, x
        )

        with pytest.raises(ValueError, match="batch_generator"):
            _ = context.batch_generator


class TestScipyCallbackAdapter:
    """Test the ScipyCallbackAdapter class."""

    def test_adapter_invokes_callbacks(self, getkey):
        """Test that callback adapter invokes callbacks correctly."""

        class RecordingCallback(klax.Callback):
            def __init__(self):
                self.started = False
                self.step_count = 0
                self.ended = False

            def on_training_start(self, context):
                self.started = True

            def on_training_step(self, context):
                self.step_count += 1
                return False

            def on_training_end(self, context):
                self.ended = True

        model = klax.nn.Linear(2, 2, uniform(), key=getkey())
        adapter = ScipyModelAdapter(model)
        x = adapter.flatten(model)

        callback = RecordingCallback()
        callback_adapter = ScipyCallbackAdapter(
            callbacks=[callback],
            adapter=adapter,
            run_state=None,
            optimizer="SLSQP",
            max_steps=100,
            loss=klax.mse,
            x=x,
        )

        callback_adapter.on_training_start()
        assert callback.started

        # Simulate training step
        from scipy.optimize import OptimizeResult

        result = OptimizeResult(x=x + 0.1, fun=0.5)
        callback_adapter.on_training_step(result)
        assert callback.step_count == 1

        callback_adapter.on_training_end()
        assert callback.ended

    def test_adapter_early_stopping(self, getkey):
        """Test that early stopping works via callbacks."""

        class StoppingCallback(klax.Callback):
            def __init__(self):
                self.call_count = 0

            def on_training_step(self, context):
                self.call_count += 1
                if self.call_count > 1:  # Stop after second call
                    return True
                return False

        model = klax.nn.Linear(2, 2, uniform(), key=getkey())
        adapter = ScipyModelAdapter(model)
        x = adapter.flatten(model)

        callback = StoppingCallback()
        callback_adapter = ScipyCallbackAdapter(
            callbacks=[callback],
            adapter=adapter,
            run_state=None,
            optimizer="SLSQP",
            max_steps=100,
            loss=klax.mse,
            x=x,
        )

        callback_adapter.on_training_start()

        from scipy.optimize import OptimizeResult

        result = OptimizeResult(x=x + 0.1, fun=0.5)

        # First step should not stop
        callback_adapter.on_training_step(result)
        assert callback_adapter.context.state.step == 1

        # Second step should raise StopIteration
        with pytest.raises(StopIteration):
            callback_adapter.on_training_step(result)


class TestScipyLossWrapper:
    """Test the scipy_loss_wrapper function."""

    def test_loss_wrapper_dtype(self, getkey):
        """Test that wrapped loss gradient is numeric."""
        model = klax.nn.Linear(2, 1, uniform(), key=getkey())
        adapter = ScipyModelAdapter(model)
        x_data = jr.normal(getkey(), (5, 2))
        y_data = jr.normal(getkey(), (5, 1))  # Match output shape

        wrapped = scipy_loss_wrapper(klax.mse, adapter, (x_data, y_data))
        flat_params = adapter.flatten(model)

        value, grad = wrapped(flat_params, run_state=None)

        # Check that gradient is numeric (not array)
        assert value.dtype == np.float64
        assert grad.dtype == np.float64
        assert isinstance(value, np.ndarray)
        assert isinstance(grad, np.ndarray)


class TestScipyFit:
    """Test the scipy_fit function."""

    def test_scipy_fit_with_callbacks(self, getkey):
        """Test scipy_fit with callbacks."""

        class CountingCallback(klax.Callback):
            def __init__(self):
                self.steps = 0

            def on_training_step(self, context):
                self.steps += 1

        x_data = jr.normal(getkey(), (10, 2))
        y_data = jnp.sum(x_data, axis=1, keepdims=True)  # Shape (10, 1)
        model = klax.nn.Linear(2, 1, uniform(), key=getkey())

        callback = CountingCallback()
        fitted_model, result = scipy_fit(
            model,
            (x_data, y_data),
            loss=klax.mse,
            optimizer="SLSQP",
            max_steps=1,
            callbacks=[callback],
        )

        # Callback should have been called
        assert callback.steps > 0

    @pytest.mark.parametrize(
        "optimizer",
        [
            "L-BFGS-B",
            "SLSQP",
        ],
    )
    def test_scipy_fit_different_optimizers(self, optimizer, getkey):
        """Test scipy_fit with different optimizers."""
        x_data = jr.normal(getkey(), (10, 2))
        y_data = jnp.sum(x_data, axis=1, keepdims=True)  # Shape (10, 1)
        model = klax.nn.Linear(2, 1, uniform(), key=getkey())
        fitted_model, result = scipy_fit(
            model,
            (x_data, y_data),
            loss=klax.mse,
            optimizer=optimizer,
            max_steps=10,
        )

    def test_scipy_fit_returns_model_type(self, getkey):
        """Test that scipy_fit returns model of correct type."""
        x_data = jr.normal(getkey(), (10, 2))
        y_data = jnp.sum(x_data, axis=1, keepdims=True)  # Shape (10, 1)
        model = klax.nn.MLP(2, 1, [8], uniform(), key=getkey())

        fitted_model, _ = scipy_fit(
            model,
            (x_data, y_data),
            loss=klax.mse,
            optimizer="SLSQP",
            max_steps=50,
        )

        # Should have same structure as input model
        assert isinstance(fitted_model, klax.nn.MLP)

    def test_scipy_fit_with_run_state(self, getkey):
        """Test scipy_fit with run_state parameter."""

        class StatefulLoss(klax.Loss):
            def value(self, model, batch, run_state):
                x, y = batch
                y_pred = jax.vmap(model)(x)
                loss = jnp.mean(jnp.square(y_pred - y))
                if run_state is not None:
                    loss = loss * run_state.get("scale", 1.0)
                return loss

        x_data = jr.normal(getkey(), (10, 2))
        y_data = jnp.sum(x_data, axis=1, keepdims=True)  # Shape (10, 1)
        model = klax.nn.Linear(2, 1, uniform(), key=getkey())

        run_state = {"scale": 2.0}

        fitted_model, result = scipy_fit(
            model,
            (x_data, y_data),
            run_state=run_state,
            loss=StatefulLoss(),
            optimizer="SLSQP",
            max_steps=1,
        )

        assert result is not None
