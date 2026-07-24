from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import klax

# ===---------------------------------------------------------------------=== #
# klax.metric
# ===---------------------------------------------------------------------=== #


class TestMetricDecorator:
    @staticmethod
    def test_attributes():
        @klax.metric(name="my_metric", verbose=True)
        def some_func(_):
            """My docstring."""
            return jnp.zeros(())

        assert some_func.name == "my_metric"
        assert some_func.verbose
        assert some_func.__doc__ == """My docstring."""


# ===---------------------------------------------------------------------=== #
# klax.BatchMetric
# ===---------------------------------------------------------------------=== #
class TestBatchMetric:
    @staticmethod
    def test_initialization(getkey):
        """Test klax.BatchMetric initialization with required parameters."""
        data = (
            jr.uniform(getkey(), (100, 2)),
            jr.uniform(getkey(), (100, 1)),
        )
        batch_size = 32
        batch_axes = 0

        metric = klax.BatchMetric(
            name="my_metric",
            func=klax.mse,
            data=data,
            batcher=klax.batch_data,
            batch_size=batch_size,
            batch_axes=batch_axes,
            verbose=False,
            jit_compile=False,
            key=getkey(),
        )

        assert metric.name == "my_metric"
        assert metric.func == klax.mse
        assert not metric.verbose

    @staticmethod
    def test_call(getkey):
        """Test __call__ method."""
        data = (
            jr.uniform(getkey(), (100, 2)),
            jr.uniform(getkey(), (100, 1)),
        )
        batch_size = 32
        batch_axes = 0

        metric = klax.BatchMetric(
            name="my_metric",
            func=klax.mse,
            data=data,
            batcher=klax.batch_data,
            batch_size=batch_size,
            batch_axes=batch_axes,
            key=getkey(),
        )

        model = eqx.nn.Linear(2, 1, key=getkey())
        context = SimpleNamespace(
            state=SimpleNamespace(model=model, run_state=None)
        )
        result = metric(context)

        assert isinstance(result, jnp.ndarray)
        assert result.shape == ()


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
    ):
        return SimpleNamespace(
            state=SimpleNamespace(
                model=model if model is not None else {"w": 1.0},
                opt_state=opt_state,
            ),
            step=step,
            steps=steps,
        )

    return _wrapped


class TestMetricLogger:
    @staticmethod
    def test_add_metric_and_logging_frequency(build_ctx):
        my_metric = klax.metric(name="m1")(lambda _: jnp.array(1.0))
        logger = klax.MetricLogger(
            log_every=2,
            metrics=[my_metric],
            verbose=0,
        )

        context = build_ctx(step=1, steps=10)

        # Step 1: not divisible by 2 -> no log
        logger.on_training_step(context)
        assert "m1" not in logger.history.content

        # Step 2: divisible by 2 -> should log
        context.step = 2
        logger.on_training_step(context)
        assert "m1" in logger.history.content
        assert logger.history.content["m1"][0] == [2]

        # Step 4: divisible by 2 -> should log again
        context.step = 4
        logger.on_training_step(context)
        assert logger.history.content["m1"][0] == [2, 4]

    @staticmethod
    def test_verbose_print_scalar_metric(capsys, build_ctx):
        logger = klax.MetricLogger(log_every=1, verbose=1)
        my_metric = klax.metric(name="loss", verbose=True)(
            lambda _: jnp.array(1.23)
        )
        logger.add_metric(my_metric)

        context = build_ctx(steps=10)
        logger.on_training_step(context)

        out = capsys.readouterr().out
        assert "Step 0/10:" in out
        # Check formatted scientific notation with 4 decimals
        assert "loss: 1.2300e+00" in out

    @staticmethod
    def test_verbose_print_non_scalar_metric(capsys, build_ctx):
        logger = klax.MetricLogger(log_every=1, verbose=1)

        my_metric = klax.metric(name="arr", verbose=True)(
            lambda _: jnp.array([1.0, 2.0])
        )
        logger.add_metric(my_metric)

        context = build_ctx(steps=5)
        logger.on_training_step(context)

        out = capsys.readouterr().out
        assert "Step 0/5:" in out
        # Representation may vary across JAX versions; just ensure it's present
        assert "arr:" in out

    @staticmethod
    def test_on_training_start_and_end_sets_history(build_ctx):
        logger = klax.MetricLogger(log_every=1, verbose=0)

        my_metric = klax.metric(name="m", verbose=False)(
            lambda _: jnp.array(0.0)
        )
        logger.add_metric(my_metric)

        sentinel_opt = {"state": 42}
        context = build_ctx(steps=7, opt_state=sentinel_opt)

        logger.on_training_start(context)
        # At start, one log happens at step 0
        assert "m" in logger.history.content
        assert logger.history.content["m"][0] == [0]

        context.step = 5
        logger.on_training_end(context)
        assert logger.history.total_steps == 5
        assert isinstance(logger.history.total_time, float)
        assert logger.history.total_time >= 0.0
        assert logger.history.final_opt_state is sentinel_opt

    @staticmethod
    def test_logger_overwrites_metric(build_ctx):
        """The logger should overwrite metrics with the same name."""
        metric_a = klax.metric(name="loss", verbose=False)(lambda _: "A")

        metric_b = klax.metric(name="loss", verbose=False)(lambda _: "B")

        logger = klax.MetricLogger(
            log_every=1, metrics=[metric_a, metric_b], verbose=0
        )
        assert len(logger.metrics) == 1

        sentinel_opt = {"state": 42}
        context = build_ctx(steps=7, opt_state=sentinel_opt)
        logger.on_training_start(context)
        assert logger.history.content["loss"][1] == ["B"]

        logger.add_metric(metric_a)
        logger.on_training_step(context)

        assert logger.history.content["loss"][1] == ["B", "A"]
