from types import SimpleNamespace
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import klax


@pytest.fixture()
def simple_metric():
    class MyMetric(klax.Metric):
        def __init__(self, name):
            super().__init__(name)

        def values(self, state):
            return {"value": jnp.array(5.0)}

    return MyMetric(name="test")


# ===---------------------------------------------------------------------=== #
# klax.Metric
# ===---------------------------------------------------------------------=== #


class TestMetric:
    def test_call(self, simple_metric):
        """Test klax.Metric returns prefixed dict."""
        state = klax.TrainingState(model=None, opt_state=None, run_state=None)

        result = simple_metric(state)

        assert isinstance(result, dict)
        assert result == {"test/value": jnp.array(5.0)}


# ===---------------------------------------------------------------------=== #
# klax.LossMetric
# ===---------------------------------------------------------------------=== #
class TestLossMetric:
    def test_call(self, getkey):
        """Test klax.LossMetric initialization with required parameters."""
        data = (
            jr.uniform(getkey(), (100, 2)),
            jr.uniform(getkey(), (100, 1)),
        )
        batch_generator = klax.batch_data(data, batch_size=32, key=getkey())

        metric = klax.LossMetric(
            loss=klax.mse,
            batch_generator=batch_generator,
            name="test",
            vmap_ensemble=False,
            jit_compile=False,
        )

        model = eqx.nn.Linear(2, 1, key=getkey())
        state = klax.TrainingState(model=model, opt_state=None, run_state=None)

        result = metric(state)

        assert isinstance(result, dict)
        assert all(k.startswith("test/") for k in result.keys())


# ===---------------------------------------------------------------------=== #
# klax.MetricLogger
# ===---------------------------------------------------------------------=== #


@pytest.fixture()
def build_ctx():
    def _wrapped(
        step: int = 0,
        steps: int = 10,
        model: dict | None = None,
        opt_state=None,
    ) -> klax.TrainingContext:
        context = SimpleNamespace(
            state=SimpleNamespace(
                model=model if model is not None else {"w": 1.0},
                opt_state=opt_state,
            ),
            step=step,
            steps=steps,
            history=klax.History(),
        )
        return cast(klax.TrainingContext, context)

    return _wrapped


class TestMetricLogger:
    def test_logging_frequency(self, build_ctx, simple_metric):

        logger = klax.MetricLogger(log_every=2, metrics=[simple_metric])

        context = build_ctx(step=1, steps=10)

        # Step 1: not divisible by 2 -> no log
        logger.on_training_step(context)
        assert "test/value" not in context.history.keys()

        # Step 2: divisible by 2 -> should log
        context.step = 2
        logger.on_training_step(context)
        assert "test/value" in context.history.keys()
        assert context.history["test/value"].steps == [2]

        # Step 4: divisible by 2 -> should log again
        context.step = 4
        logger.on_training_step(context)
        assert context.history["test/value"].steps == [2, 4]

    def test_on_training_start_and_end_sets_history(
        self, build_ctx, simple_metric
    ):

        logger = klax.MetricLogger(log_every=1, metrics=[simple_metric])

        context = build_ctx(steps=7)

        logger.on_training_start(context)
        # At start, one log happens at step 0
        assert "test/value" in context.history.keys()
        assert context.history["test/value"][0] == [0]

        context.step = 5
        logger.on_training_end(context)
        assert context.history.total_steps == 5
        assert isinstance(context.history.total_time, float)
        assert context.history.total_time >= 0.0

    def test_duplicate_namespaces_raises_error(self, simple_metric):
        with pytest.raises(ValueError):
            klax.MetricLogger(
                log_every=1, metrics=[simple_metric, simple_metric]
            )
