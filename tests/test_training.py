import equinox as eqx
import klax
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array
import optax

from klax.callbacks import Callback, CallbackArgs, HistoryCallback

def test_training(getkey):
    # Fitting a linear function
    x = jnp.linspace(0.0, 1.0, 2).reshape(-1, 1)
    y = 2.0 * x + 1.0
    model = eqx.nn.Linear(1, 1, key=getkey())
    model, _ = klax.fit(model, (x, y), optimizer=optax.adam(1.0), key=getkey())
    y_pred = jax.vmap(model)(x)
    assert jnp.allclose(y_pred, y)

    # Multiple inputs
    class Model(eqx.Module):
        weight: Array

        def __call__(self, x):
            b, x = x
            return b + self.weight * x

    x = jrandom.uniform(key=getkey(), shape=(10,))
    b = 2.0
    y = b + 2 * x
    model = Model(weight=jnp.array(1.0))
    model, _ = klax.fit(
        model,
        ((b, x), y),
        batch_axis=0,  # Test automatic batch axis braodcasting to data
        optimizer=optax.adam(1.0),
        key=getkey(),
    )
    y_pred = jax.vmap(model, in_axes=((None, 0),))((b, x))
    assert jnp.allclose(y_pred, y)

    # History shape
    x = jrandom.uniform(getkey(), (2, 1))
    model = eqx.nn.Linear(1, 1, key=getkey())
    history = HistoryCallback(log_every=100)
    model, history = klax.fit(model, (x, x), steps=1000, history=history, key=getkey())
    assert len(history.steps) == 11
    assert len(history.loss) == 11
    time_1 = history.training_time
    model, history = klax.fit(model, (x, x), steps=500, history=history, key=getkey())
    assert len(history.steps) == 16
    assert len(history.loss) == 16
    assert history.steps[-1] == 1500
    time_2 = history.training_time
    assert time_1 < time_2


    # Validation data
    x = jrandom.uniform(getkey(), (2, 1))
    model = eqx.nn.Linear(1, 1, key=getkey())
    _, history = klax.fit(model, (x, x), validation_data=(x, x), key=getkey())
    assert len(history.val_loss) == 11

    # Callbacks
    x = jrandom.uniform(getkey(), (2, 1))
    model = eqx.nn.Linear(1, 1, key=getkey())

    class MyCallback(Callback):
        def __call__(self, cbargs: CallbackArgs):
            if cbargs.step == 123:
                return True

    _, history = klax.fit(
        model, (x, x), history=HistoryCallback(1), callbacks=(MyCallback(),), key=getkey()
    )
    print(history.log_every)
    assert history.steps[-1] == 123


    # Test all optax optimizers
    optimizers = [
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
    ]
    x = jnp.linspace(0.0, 1.0, 2)[:, None]
    y = 2.0 * x + 1.0
    for optimizer in optimizers:
        model = eqx.nn.MLP(1, 1, 64, 2, key=getkey())
        try:
            model, _ = klax.fit(model, (x, y), optimizer=optimizer, key=getkey())
        except Exception as e:
            print(f"Optimizer {optimizer} failed with error: {e}")
            raise e

