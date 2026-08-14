import jax.numpy as jnp
import pytest

import klax


class TestHistory:
    def test_append_metric(self):
        """Test appending a single metric entry."""
        history = klax.History()
        history.append(key="loss", step=0, value=jnp.array(0.5))
        history.append(key="loss", step=1, value=jnp.array(0.2))
        history.append(key="array", step=1, value=jnp.array([0.2, 2.0]))

        assert "loss" in history.content
        assert history.content["loss"].steps == [0, 1]
        assert history.content["loss"].values == [
            jnp.array(0.5),
            jnp.array(0.2),
        ]

        assert history.content["array"].steps == [1]
        assert jnp.all(
            history.content["array"].values[0] == jnp.array([0.2, 2.0])
        )

    def test_getitem_existing_metric(self):
        """Test retrieving an existing metric using __getitem__."""
        history = klax.History()
        history.append(key="loss", step=0, value=jnp.array(0.5))
        history.append(key="loss", step=10, value=jnp.array(0.3))

        steps, values = history["loss"]
        assert steps == [0, 10]
        assert values == [jnp.array(0.5), jnp.array(0.3)]

    def test_keys_returns_all_metric_keys(self):
        """Test that keys() returns all metric keys."""
        history = klax.History()
        history.append(key="loss", step=0, value=0.5)
        history.append(key="accuracy", step=0, value=0.8)

        assert set(history.keys()) == {"loss", "accuracy"}

    def test_getitem_nonexistent_metric_raises_keyerror(self):
        """Test that accessing non-existent metric raises KeyError."""
        history = klax.History()

        with pytest.raises(KeyError, match="Metric 'accuracy' not found"):
            history["accuracy"]

    def test_extend(self):
        """Test extending one klax.History with another."""
        history1 = klax.History()
        history1.append(key="loss", step=0, value=jnp.array(0.5))
        history1.append(key="loss", step=10, value=jnp.array(0.3))
        history1.append(key="hist1_metric", step=10, value=jnp.array(-5))
        history1.total_steps = 15
        history1.total_time = 4.2

        history2 = klax.History()
        history2.append(key="loss", step=0, value=jnp.array(0.2))
        history2.append(key="loss", step=10, value=jnp.array(0.1))
        history2.append(key="hist2_metric", step=10, value=jnp.array(5))
        history2.total_steps = 11
        history2.total_time = 3.0

        history1.extend(history2)

        steps, values = history1["loss"]
        assert steps == [0, 10, 15, 25]
        assert values == [
            jnp.array(0.5),
            jnp.array(0.3),
            jnp.array(0.2),
            jnp.array(0.1),
        ]

        steps, values = history1["hist1_metric"]
        assert steps == [10]
        assert values == [jnp.array(-5)]

        steps, values = history1["hist2_metric"]
        assert steps == [25]
        assert values == [jnp.array(5)]

        assert history1.total_steps == 26
        assert history1.total_time == 7.2

    def test_save_and_load_roundtrip(self, tmp_path):
        history = klax.History()
        history.append(key="loss", step=0, value=jnp.array(0.5))
        history.append(key="acc", step=5, value=jnp.array(0.8))
        history.total_steps = 5
        history.total_time = 1.25

        path = tmp_path / "nested" / "history.pkl"

        history.save(path)
        loaded = klax.History.load(path)

        assert loaded.content == history.content
        assert loaded.total_steps == history.total_steps
        assert loaded.total_time == history.total_time
