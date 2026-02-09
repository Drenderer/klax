import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import klax


class TestMakeStep:
    @pytest.mark.parametrize(
        "optimizer",
        [
            optax.adabelief(1.0),
            optax.adadelta(1.0),
            optax.adan(1.0),
            optax.adafactor(1.0),
            optax.adagrad(1.0),
            optax.adam(1.0),
            optax.adamw(1.0),
            optax.adamax(1.0),
            optax.adamaxw(1.0),
            optax.amsgrad(1.0),
            optax.fromage(1.0),
            optax.lamb(1.0),
            optax.lars(1.0),
            optax.lbfgs(1.0),
            optax.lion(1.0),
            optax.nadam(1.0),
            optax.nadamw(1.0),
            optax.noisy_sgd(1.0),
            optax.novograd(1.0),
            optax.optimistic_gradient_descent(1.0),
            optax.optimistic_adam(1.0),
            optax.polyak_sgd(1.0),
            optax.radam(1.0),
            optax.rmsprop(1.0),
            optax.sgd(1.0),
            optax.sign_sgd(1.0),
            optax.sm3(1.0),
            optax.yogi(1.0),
        ],
    )
    def test_make_step_updates_state(self, optimizer, getkey):
        model = klax.nn.FICNN(2, "scalar", [4, 4], key=getkey())
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([3.0, 7.0])
        batch = klax.batch_data((x, y), batch_size=32, key=getkey())

        state, static = klax.make_state_and_static(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            batch=batch,
            batch_axes=0,
            loss=klax.mse,
            steps=5,
        )

        new_state = klax.make_step(state, next(static.batch), static)

        # State has changed
        assert not jax.tree.all(
            jax.tree.map(
                lambda a, b: jnp.array_equal(a, b)
                if isinstance(a, jnp.ndarray)
                else a == b,
                state.model_leaves,
                new_state.model_leaves,
            )
        )


class TestRunTrainingLoop:
    def test_run_training_loop_invokes_callbacks(self, getkey):
        class RecordingCallback(klax.Callback):
            def __init__(self):
                self.start_steps = []
                self.steps = []
                self.end_steps = []

            def on_training_start(self, view, step):
                self.start_steps.append(step)

            def on_training_step(self, view, step):
                self.steps.append(step)

            def on_training_end(self, view, step):
                self.end_steps.append(step)

        model = klax.nn.FICNN(2, "scalar", [4, 4], key=getkey())
        optimizer = optax.sgd(1.0)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([3.0, 7.0])
        batch = klax.batch_data((x, y), batch_size=32, key=getkey())

        state, static = klax.make_state_and_static(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            batch=batch,
            batch_axes=0,
            loss=klax.mse,
            steps=3,
        )

        callback = RecordingCallback()

        new_state = klax.run_training_loop(state, static, [callback])

        assert callback.start_steps == [0]
        assert callback.steps == [1, 2, 3]
        assert callback.end_steps == [3]
        assert not jax.tree.all(
            jax.tree.map(
                lambda a, b: jnp.array_equal(a, b)
                if isinstance(a, jnp.ndarray)
                else a == b,
                state.model_leaves,
                new_state.model_leaves,
            )
        )

    def test_run_training_loop_zero_steps_no_updates(self, getkey):
        class RecordingCallback(klax.Callback):
            def __init__(self):
                self.start_steps = []
                self.steps = []
                self.end_steps = []

            def on_training_start(self, view, step):
                self.start_steps.append(step)

            def __call__(self, view, step):
                self.steps.append(step)

            def on_training_end(self, view, step):
                self.end_steps.append(step)

        model = klax.nn.FICNN(2, "scalar", [4, 4], key=getkey())
        optimizer = optax.sgd(1.0)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([3.0, 7.0])
        batch = klax.batch_data((x, y), batch_size=32, key=getkey())

        state, static = klax.make_state_and_static(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            batch=batch,
            batch_axes=0,
            loss=klax.mse,
            steps=0,
        )

        callback = RecordingCallback()

        new_state = klax.run_training_loop(state, static, [callback])

        assert callback.start_steps == [0]
        assert callback.steps == []
        assert callback.end_steps == [0]
        assert jax.tree.all(
            jax.tree.map(
                lambda a, b: jnp.array_equal(a, b)
                if isinstance(a, jnp.ndarray)
                else a == b,
                state.model_leaves,
                new_state.model_leaves,
            )
        )

    def test_run_training_loop_stops_on_callback(self, getkey):
        class StopAfterOne(klax.Callback):
            def __init__(self):
                self.steps = []
                self.end_steps = []

            def on_training_step(self, view, step):
                self.steps.append(step)
                return True  # request stop after first step

            def on_training_end(self, view, step):
                self.end_steps.append(step)

        model = klax.nn.FICNN(2, "scalar", [4, 4], key=getkey())
        optimizer = optax.sgd(1.0)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([3.0, 7.0])
        batch = klax.batch_data((x, y), batch_size=32, key=getkey())

        state, static = klax.make_state_and_static(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            batch=batch,
            batch_axes=0,
            loss=klax.mse,
            steps=5,
        )

        callback = StopAfterOne()

        new_state = klax.run_training_loop(state, static, [callback])

        assert callback.steps == [1]
        assert callback.end_steps == [1]
        assert not jax.tree.all(
            jax.tree.map(
                lambda a, b: jnp.array_equal(a, b)
                if isinstance(a, jnp.ndarray)
                else a == b,
                state.model_leaves,
                new_state.model_leaves,
            )
        )


class TestFit:
    def test_fit_returns_history_with_loss(self, getkey):
        model = klax.nn.FICNN(2, "scalar", [4, 4], key=getkey())

        data = (
            jr.uniform(getkey(), (100, 2)),
            jr.uniform(getkey(), (100,)),
        )

        trained_model, history = klax.fit(
            model,
            data,
            batch_size=5,
            batch_axes=0,
            steps=5,
            loss=klax.mse,
            optimizer=optax.sgd(0.1),
            key=getkey(),
        )

        assert isinstance(trained_model, klax.nn.FICNN)
        assert history.total_steps == 5
        assert "loss" in history.content
        loss_steps, loss_values = history["loss"]
        assert loss_steps == [0]
        assert len(loss_values) == 1

    def test_fit_without_logger_returns_empty_history(self, getkey):
        model = klax.nn.FICNN(2, "scalar", [4, 4], key=getkey())

        data = (
            jr.uniform(getkey(), (100, 2)),
            jr.uniform(getkey(), (100,)),
        )
        trained_model, history = klax.fit(
            model,
            data,
            batch_size=5,
            batch_axes=0,
            steps=5,
            loss=klax.mse,
            optimizer=optax.sgd(0.1),
            logger=None,
            key=getkey(),
        )
        initial_leaves = eqx.filter(model, eqx.is_inexact_array)

        assert history.total_steps == -1
        assert history.content == {}
        updated_leaves = eqx.filter(trained_model, eqx.is_inexact_array)
        assert not jax.tree.all(
            jax.tree.map(
                lambda a, b: jnp.array_equal(a, b)
                if isinstance(a, jnp.ndarray)
                else a == b,
                initial_leaves,
                updated_leaves,
            )
        )
