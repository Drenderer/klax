import equinox as eqx
import klax
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array
import numpy as np
import optax
import pytest


def test_dataloader(getkey):
    # Sequence with one element
    x = jrandom.uniform(getkey(), (10,))
    data = (x,)
    generator = klax.dataloader(data, key=getkey())
    assert isinstance(next(generator), tuple)
    assert len(next(generator)) == 1

    # Nested PyTree
    x = jrandom.uniform(getkey(), (10,))
    data = [x, (x, {"a": x, "b": x})]
    generator = klax.dataloader(data, key=getkey())
    assert isinstance(next(generator), list)
    assert len(next(generator)) == 2
    assert isinstance(next(generator)[1], tuple)
    assert len(next(generator)[1]) == 2
    assert isinstance(next(generator)[1][1], dict)
    assert len(next(generator)[1][1]) == 2

    # Default batch size
    x = jrandom.uniform(getkey(), (33,))
    data = (x,)
    generator = klax.dataloader(data, key=getkey())
    assert next(generator)[0].shape[0] == 32

    # Batch mask
    x = jrandom.uniform(getkey(), (10,))
    data = (x, (x, x))
    batch_mask = (True, (False, True))
    generator = klax.dataloader(data, 2, batch_mask, key=getkey())
    assert next(generator)[0].shape[0] == 2
    assert next(generator)[1][0].shape[0] == 10
    assert next(generator)[1][1].shape[0] == 2

    # Different PyTree structures
    x = jrandom.uniform(getkey(), (10,))
    data = (x, x)
    batch_mask = (True,)
    with pytest.raises(
        ValueError,
        match="Arguments data and batch_mask must have equal PyTree structures.",
    ):
        generator = klax.dataloader(data, batch_mask=batch_mask, key=getkey())
        next(generator)

    # No batch dimensions
    x = jrandom.uniform(getkey(), (10,))
    data = (x,)
    batch_mask = (False,)
    with pytest.raises(
        ValueError, match="At least one element must have batch dimension."
    ):
        generator = klax.dataloader(data, batch_mask=batch_mask, key=getkey())
        next(generator)

    # Different batch dimensions
    x = jrandom.uniform(getkey(), (10,))
    y = jrandom.uniform(getkey(), (5,))
    data = (x, y)
    with pytest.raises(
        ValueError, match="All batched arrays must have equal batch dimension."
    ):
        generator = klax.dataloader(data, key=getkey())
        next(generator)


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
    b = jnp.array(2.0)
    y = b + 2 * x
    model = Model(weight=jnp.array(1.0))
    model, _ = klax.fit(
        model,
        ((b, x), y),
        data_mask=((False, True), True),
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

    def callback(cbargs: klax.callbacks.CallbackArgs):
        if cbargs.step == 123:
            return True

    _, history = klax.fit(
        model, (x, x), log_every=1, callbacks=[callback], key=getkey()
    )
    assert history.steps[-1] == 123
