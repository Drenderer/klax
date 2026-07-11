import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import klax

# ===---------------------------------------------------------------------=== #
# klax.make_step
# ===---------------------------------------------------------------------=== #


class TestMakeStep:
    @staticmethod
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
            optax.noisy_sgd(1.0, key=jr.key(0)),
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
    def test_make_step_updates_state(optimizer, getkey):
        model = klax.nn.FICNN(2, "scalar", [4, 4], key=getkey())
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([3.0, 7.0])

        state = klax.TrainingState(model, opt_state, run_state=None)
        state_leaves, state_treedef = jax.tree.flatten(state)

        new_state_leaves = klax.make_step(
            state_leaves,
            state_treedef,
            batch=(x, y),
            loss=klax.mse,
            optimizer=optimizer,
        )

        # State has changed
        assert not jax.tree.all(
            jax.tree.map(
                lambda a, b: jnp.array_equal(a, b)
                if isinstance(a, jnp.ndarray)
                else a == b,
                state_leaves,
                new_state_leaves,
            )
        )


# ===---------------------------------------------------------------------=== #
# klax.run_training_loop
# ===---------------------------------------------------------------------=== #


class TestRunTrainingLoop:
    @staticmethod
    def test_invokes_callbacks(getkey):
        class RecordingCallback(klax.Callback):
            def __init__(self):
                self.start_steps = []
                self.steps = []
                self.end_steps = []

            def on_training_start(self, context):
                self.start_steps.append(context.step)

            def on_training_step(self, context):
                self.steps.append(context.step)

            def on_training_end(self, context):
                self.end_steps.append(context.step)

        model = klax.nn.FICNN(2, "scalar", [4, 4], key=getkey())
        optimizer = optax.sgd(1.0)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([3.0, 7.0])
        batch_generator = klax.batch_data((x, y), batch_size=32, key=getkey())

        context = klax.TrainingContext(
            model,
            optimizer,
            opt_state,
            batch_generator,
            run_state=None,
            loss=klax.mse,
            steps=3,
        )

        callback = RecordingCallback()

        context = klax.run_training_loop(
            context, [callback], step_function=klax.make_step
        )

        assert callback.start_steps == [0]
        assert callback.steps == [1, 2, 3]
        assert callback.end_steps == [3]
        assert not jax.tree.all(
            jax.tree.map(
                lambda a, b: jnp.array_equal(a, b)
                if isinstance(a, jnp.ndarray)
                else a == b,
                model,
                context.state.model,
            )
        )

    @staticmethod
    def test_zero_steps_no_updates(getkey):
        class RecordingCallback(klax.Callback):
            def __init__(self):
                self.start_steps = []
                self.steps = []
                self.end_steps = []

            def on_training_start(self, context):
                self.start_steps.append(context.step)

            def on_training_step(self, context):
                self.steps.append(context.step)

            def on_training_end(self, context):
                self.end_steps.append(context.step)

        model = klax.nn.FICNN(2, "scalar", [4, 4], key=getkey())
        optimizer = optax.sgd(1.0)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([3.0, 7.0])
        batch_generator = klax.batch_data((x, y), batch_size=32, key=getkey())

        context = klax.TrainingContext(
            model,
            optimizer,
            opt_state,
            batch_generator,
            run_state=None,
            loss=klax.mse,
            steps=0,
        )

        callback = RecordingCallback()

        updated_view = klax.run_training_loop(
            context, [callback], step_function=klax.make_step
        )

        assert callback.start_steps == [0]
        assert callback.steps == []
        assert callback.end_steps == [0]
        assert jax.tree.all(
            jax.tree.map(
                lambda a, b: jnp.array_equal(a, b)
                if isinstance(a, jnp.ndarray)
                else a == b,
                model,
                context.state.model,
            )
        )

    @staticmethod
    def test_stops_on_callback(getkey):
        class StopAfterOne(klax.Callback):
            def __init__(self):
                self.steps = []
                self.end_steps = []

            def on_training_step(self, context):
                self.steps.append(context.step)
                return True  # request stop after first step

            def on_training_end(self, context):
                self.end_steps.append(context.step)

        model = klax.nn.FICNN(2, "scalar", [4, 4], key=getkey())
        optimizer = optax.sgd(1.0)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([3.0, 7.0])
        batch_generator = klax.batch_data((x, y), batch_size=32, key=getkey())

        context = klax.TrainingContext(
            model,
            optimizer,
            opt_state,
            batch_generator,
            run_state=None,
            loss=klax.mse,
            steps=10,
        )

        callback = StopAfterOne()

        context = klax.run_training_loop(
            context, [callback], step_function=klax.make_step
        )

        assert callback.steps == [1]
        assert callback.end_steps == [1]
        assert not jax.tree.all(
            jax.tree.map(
                lambda a, b: jnp.array_equal(a, b)
                if isinstance(a, jnp.ndarray)
                else a == b,
                model,
                context.state.model,
            )
        )


# ===---------------------------------------------------------------------=== #
# klax.fit
# ===---------------------------------------------------------------------=== #


class TestFit:
    @staticmethod
    def test_fit(getkey):
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

    @staticmethod
    def test_fit_with_xarray(getkey):
        xr = pytest.importorskip(
            "xarray", reason="xarray dependency is not installed"
        )
        # xarray_jax registers xr.Dataset/DataArray as JAX pytrees, which is
        # required for `eqx.filter_jit` to flatten an xarray batch into array
        # leaves instead of trying to hash the whole Dataset as a static arg.
        pytest.importorskip(
            "xarray_jax", reason="xarray_jax dependency is not installed"
        )

        model = klax.nn.FICNN(2, "scalar", [4, 4], key=getkey())

        class MyLoss(klax.Loss):
            def value(self, model, batch, run_state):
                y_pred = jax.vmap(model)(batch.x.data)
                return jnp.mean((batch.y.data - y_pred) ** 2)

        data = xr.Dataset(
            {
                "x": (("batch", "i"), jr.uniform(getkey(), (100, 2))),
                "y": (("batch",), jr.uniform(getkey(), (100,))),
            },
            coords={"batch": jnp.arange(100), "i": jnp.arange(2)},
        )

        trained_model, history = klax.fit(
            model,
            data,
            batch_size=5,
            batch_axes="batch",
            steps=5,
            loss=MyLoss(),
            optimizer=optax.sgd(0.1),
            key=getkey(),
        )

        assert isinstance(trained_model, klax.nn.FICNN)
        assert history.total_steps == 5
        assert "loss" in history.content
        loss_steps, loss_values = history["loss"]
        assert loss_steps == [0]
        assert len(loss_values) == 1

    @staticmethod
    def test_overwriting_default_metrics(getkey):
        model = klax.nn.FICNN(2, "scalar", [4, 4], key=getkey())

        data = (
            jr.uniform(getkey(), (100, 2)),
            jr.uniform(getkey(), (100,)),
        )

        def my_metric(model):
            return "A"

        my_metric.name = "loss"
        my_metric.verbose = False

        _, history = klax.fit(
            model,
            data,
            batch_size=5,
            batch_axes=0,
            steps=5,
            loss=klax.mse,
            optimizer=optax.sgd(0.1),
            metrics=[my_metric],
            key=getkey(),
        )

        assert history.content["loss"][1] == ["A"]

    def test_vmap_training(self, getkey):
        # Create ensemble of models
        keys = jr.split(getkey(), 3)

        @eqx.filter_vmap
        def make_ensemble(key):
            return klax.nn.FICNN("scalar", "scalar", [4, 4], key=key)

        model_ensemble = make_ensemble(keys)

        # Create simple data
        x = jr.uniform(getkey(), (20,))
        y = jr.uniform(getkey(), (20,))

        trained_ensemble, history = klax.fit(
            model_ensemble,
            (x, y),
            batch_size=5,
            steps=4,
            loss=klax.mse,
            optimizer=optax.sgd(0.1),
            vmap_ensemble=True,
            key=getkey(),
        )
