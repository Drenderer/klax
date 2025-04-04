import equinox as eqx
import klax
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array
import optax

from klax.callbacks import CallbackArgs, HistoryCallback

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

    # History shape and type
    x = jrandom.uniform(getkey(), (2, 1))
    model = eqx.nn.Linear(1, 1, key=getkey())
    _, history = klax.fit(model, (x, x), key=getkey())
    assert len(history.steps) == 10
    assert len(history.loss) == 10
    assert isinstance(history.training_time, float)

    # Validation data
    x = jrandom.uniform(getkey(), (2, 1))
    model = eqx.nn.Linear(1, 1, key=getkey())
    _, history = klax.fit(model, (x, x), validation_data=(x, x), key=getkey())
    assert len(history.val_loss) == 10

    # Callbacks
    x = jrandom.uniform(getkey(), (2, 1))
    model = eqx.nn.Linear(1, 1, key=getkey())

    def callback(cbargs: CallbackArgs):
        if cbargs.step == 123:
            return True

    _, history = klax.fit(
        model, (x, x), history=HistoryCallback(1), callbacks=(callback,), key=getkey()
    )
    print(history.log_every)
    assert history.steps[-1] == 123
