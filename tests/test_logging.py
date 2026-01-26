from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import klax
from klax import History, LossMetric, MetricLogger


class TestLossMetric:
    """Test suite for the LossMetric class."""

    def test_initialization(self, getkey):
        """Test LossMetric initialization with required parameters."""
        data = (
            jr.uniform(getkey(), (100, 2)),
            jr.uniform(getkey(), (100, 1)),
        )
        batch_size = 32
        batch_axes = 0

        metric = LossMetric(
            batcher=klax.batch_data,
            data=data,
            batch_size=batch_size,
            batch_axes=batch_axes,
            loss=klax.mse,
            key=getkey(),
        )

        assert metric.batch_axes == batch_axes
        assert metric.loss == klax.mse

    def test_call(self, getkey):
        """Test __call__ method."""
        data = (
            jr.uniform(getkey(), (100, 2)),
            jr.uniform(getkey(), (100, 1)),
        )
        batch_size = 32
        batch_axes = 0

        metric = LossMetric(
            batcher=klax.batch_data,
            data=data,
            batch_size=batch_size,
            batch_axes=batch_axes,
            loss=klax.mse,
            key=getkey(),
        )

        model = eqx.nn.Linear(2, 1, key=getkey())
        result = metric(model)

        assert isinstance(result, jnp.ndarray)
        assert result.shape == ()


class TestHistory:
    """Test suite for the History class."""

    def test_initialization(self):
        """Test History initialization."""
        history = History()
        assert history.content == {}
        assert history.total_time == -1.0
        assert history.total_steps == -1
        assert history.final_opt_state is None

    def test_append_single_metric(self):
        """Test appending a single metric entry."""
        history = History()
        history.append(step=0, key="loss", value=0.5)

        assert "loss" in history.content
        assert history.content["loss"][0] == [0]
        assert history.content["loss"][1] == [0.5]

    def test_append_multiple_entries_same_metric(self):
        """Test appending multiple entries to the same metric."""
        history = History()
        history.append(step=0, key="loss", value=0.5)
        history.append(step=10, key="loss", value=0.3)
        history.append(step=20, key="loss", value=0.1)

        steps, values = history.content["loss"]
        assert steps == [0, 10, 20]
        assert values == [0.5, 0.3, 0.1]

    def test_append_multiple_different_metrics(self):
        """Test appending entries to different metrics."""
        history = History()
        history.append(step=0, key="loss", value=0.5)
        history.append(step=0, key="accuracy", value=0.8)
        history.append(step=10, key="loss", value=0.3)
        history.append(step=10, key="accuracy", value=0.85)

        assert "loss" in history.content
        assert "accuracy" in history.content
        assert history.content["loss"][0] == [0, 10]
        assert history.content["accuracy"][0] == [0, 10]

    def test_getitem_existing_metric(self):
        """Test retrieving an existing metric using __getitem__."""
        history = History()
        history.append(step=0, key="loss", value=0.5)
        history.append(step=10, key="loss", value=0.3)

        steps, values = history["loss"]
        assert steps == [0, 10]
        assert values == [0.5, 0.3]

    def test_getitem_nonexistent_metric_raises_keyerror(self):
        """Test that accessing non-existent metric raises KeyError."""
        history = History()
        history.append(step=0, key="loss", value=0.5)

        with pytest.raises(KeyError, match="Metric 'accuracy' not found"):
            history["accuracy"]

    def test_append_with_various_value_types(self):
        """Test appending metrics with various value types."""
        history = History()

        # Float value
        history.append(step=0, key="loss", value=0.5)
        # Integer value
        history.append(step=1, key="epoch", value=1)
        # Array value
        history.append(step=2, key="gradients", value=jnp.array([0.1, 0.2]))
        # Complex value
        history.append(step=3, key="dict_metric", value={"a": 1, "b": 2})

        assert history.content["loss"][1][0] == 0.5
        assert history.content["epoch"][1][0] == 1
        assert jnp.allclose(
            history.content["gradients"][1][0], jnp.array([0.1, 0.2])
        )
        assert history.content["dict_metric"][1][0] == {"a": 1, "b": 2}

    def test_append_maintains_order(self):
        """Test that append maintains insertion order."""
        history = History()
        for i in range(100):
            history.append(step=i, key="loss", value=float(i))

        steps, values = history["loss"]
        assert steps == list(range(100))
        assert values == [float(i) for i in range(100)]

    def test_content_structure(self):
        """Test the structure of the content dictionary."""
        history = History()
        history.append(step=0, key="metric1", value=1.0)
        history.append(step=5, key="metric1", value=2.0)

        # Each metric should map to a tuple of (steps, values)
        metric_data = history.content["metric1"]
        assert isinstance(metric_data, tuple)
        assert len(metric_data) == 2
        assert isinstance(metric_data[0], list)
        assert isinstance(metric_data[1], list)

    def test_extend(self):
        """Test extending one History with another."""
        history1 = History(total_steps=10, total_time=5.0, final_opt_state=1)
        history1.append(step=0, key="loss", value=0.5)
        history1.append(step=10, key="loss", value=0.3)
        history1.append(step=10, key="hist1_metric", value=-5)

        history2 = History(total_steps=15, total_time=3.0, final_opt_state=2)
        history2.append(step=0, key="loss", value=0.2)
        history2.append(step=10, key="loss", value=0.1)
        history2.append(step=10, key="hist2_metric", value=5)

        history1.extend(history2)

        steps, values = history1["loss"]
        assert steps == [0, 10, 10, 20]
        assert values == [0.5, 0.3, 0.2, 0.1]

        steps, values = history1["hist1_metric"]
        assert steps == [10]
        assert values == [-5]

        steps, values = history1["hist2_metric"]
        assert steps == [20]
        assert values == [5]

        assert history1.total_steps == 25
        assert history1.total_time == 8.0
        assert history1.final_opt_state == 2

    def test_save_and_load_roundtrip(self, tmp_path):
        history = History()
        history.append(step=0, key="loss", value=0.5)
        history.append(step=5, key="acc", value=0.8)
        history.total_steps = 5
        history.total_time = 1.25
        history.final_opt_state = {"opt": 1}

        path = tmp_path / "nested" / "history.pkl"
        history.save(path)

        loaded = History.load(path)

        assert loaded.content == history.content
        assert loaded.total_steps == history.total_steps
        assert loaded.total_time == history.total_time
        assert loaded.final_opt_state == history.final_opt_state


class TestMetricLogger:
    """Tests for the MetricLogger class."""

    def _make_view(
        self,
        steps: int = 10,
        model: dict | None = None,
        opt_state: object | None = None,
    ):
        return SimpleNamespace(
            model=model if model is not None else {"w": 1.0},
            static=SimpleNamespace(steps=steps),
            opt_state=opt_state if opt_state is not None else {"opt": 1},
        )

    def test_add_metric_and_logging_frequency(self):
        logger = MetricLogger(log_every=2, verbose=False)
        logger.add_metric("m1", lambda model: jnp.array(1.0), verbose=False)

        view = self._make_view(steps=10)

        # Step 1: not divisible by 2 -> no log
        logger(view, 1)
        assert "m1" not in logger.history.content

        # Step 2: divisible by 2 -> should log
        logger(view, 2)
        assert "m1" in logger.history.content
        assert logger.history.content["m1"][0] == [2]

        # Step 4: divisible by 2 -> should log again
        logger(view, 4)
        assert logger.history.content["m1"][0] == [2, 4]

    def test_verbose_print_scalar_metric(self, capsys):
        logger = MetricLogger(log_every=1, verbose=True)
        logger.add_metric("loss", lambda model: jnp.array(1.23), verbose=True)

        view = self._make_view(steps=10)
        logger(view, 0)

        out = capsys.readouterr().out
        assert "Step 0/10:" in out
        # Check formatted scientific notation with 4 decimals
        assert "loss: 1.2300e+00" in out

    def test_verbose_print_non_scalar_metric(self, capsys):
        logger = MetricLogger(log_every=1, verbose=True)
        logger.add_metric(
            "arr", lambda model: jnp.array([1.0, 2.0]), verbose=True
        )

        view = self._make_view(steps=5)
        logger(view, 0)

        out = capsys.readouterr().out
        assert "Step 0/5:" in out
        # Representation may vary across JAX versions; just ensure it's present
        assert "arr:" in out

    def test_on_training_start_and_end_sets_history(self):
        logger = MetricLogger(log_every=1, verbose=False)
        logger.add_metric("m", lambda model: jnp.array(0.0), verbose=False)

        sentinel_opt = {"state": 42}
        view = self._make_view(steps=7, opt_state=sentinel_opt)

        logger.on_training_start(view, 0)
        # At start, one log happens at step 0
        assert "m" in logger.history.content
        assert logger.history.content["m"][0] == [0]

        logger.on_training_end(view, 5)
        assert logger.history.total_steps == 5
        assert isinstance(logger.history.total_time, float)
        assert logger.history.total_time >= 0.0
        assert logger.history.final_opt_state is sentinel_opt

    def test_add_metric_registers_with_verbose_flag(self):
        logger = MetricLogger(log_every=10, verbose=False)
        logger.add_metric("acc", lambda model: jnp.array(0.9), verbose=True)

        assert "acc" in logger.metric_defs
        verbose_flag, metric_fn = logger.metric_defs["acc"]
        assert verbose_flag is True
        assert callable(metric_fn)
