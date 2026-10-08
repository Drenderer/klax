from types import SimpleNamespace
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import klax


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
            prefix="test",
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
    @staticmethod
    def test_add_metric_and_logging_frequency(build_ctx):

        def my_metric(state):
            return {"m1": jnp.array(1.0)}

        logger = klax.MetricLogger(log_every=2, metrics=[my_metric])

        context = build_ctx(step=1, steps=10)

        # Step 1: not divisible by 2 -> no log
        logger.on_training_step(context)
        assert "m1" not in context.history.keys()

        # Step 2: divisible by 2 -> should log
        context.step = 2
        logger.on_training_step(context)
        assert "m1" in context.history.keys()
        assert context.history["m1"].steps == [2]

        # Step 4: divisible by 2 -> should log again
        context.step = 4
        logger.on_training_step(context)
        assert context.history["m1"].steps == [2, 4]

    @staticmethod
    def test_on_training_start_and_end_sets_history(build_ctx):

        def my_metric(state):
            return {"m": jnp.array(1.0)}

        logger = klax.MetricLogger(log_every=1, metrics=[my_metric])

        context = build_ctx(steps=7)

        logger.on_training_start(context)
        # At start, one log happens at step 0
        assert "m" in context.history.keys()
        assert context.history["m"][0] == [0]

        context.step = 5
        logger.on_training_end(context)
        assert context.history.total_steps == 5
        assert isinstance(context.history.total_time, float)
        assert context.history.total_time >= 0.0
