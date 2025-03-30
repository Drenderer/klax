import equinox as eqx
import jax
import jax.numpy as jnp

from klax.callbacks import CallbackArgs, DefaultHistoryCallback


def test_callbackargs():
    # Test lazy evaluation of loss function
    counter = 0

    def get_loss(model, data):
        x, y = data
        nonlocal counter
        counter += 1
        y_pred = jax.vmap(model)(x)
        return jnp.mean(jnp.square(y - y_pred))

    class Model(eqx.Module):
        def __call__(self, x):
            return 1.1 * x

    data = 2 * (jnp.array([1, 2, 3, 4, 5]),)
    val_data = 2 * (jnp.array([6, 7, 8]),)

    model = Model()
    flat_model, treedef_model = jax.tree_util.tree_flatten(model)

    cbargs = CallbackArgs(get_loss, treedef_model, data, val_data)
    cbargs.update(flat_model, 1)

    assert counter == 0
    loss_1 = cbargs.loss
    loss_2 = cbargs.loss
    assert loss_1 == loss_2
    assert counter == 1
    _ = cbargs.val_loss
    assert counter == 2
    cbargs.update(flat_model, 2)
    _ = cbargs.loss
    assert counter == 3


def test_default_history_callback():
    # >>> Just define stuff for cbargs
    def get_loss(model, data):
        x, y = data
        y_pred = jax.vmap(model)(x)
        return jnp.mean(jnp.square(y - y_pred))

    data = 2 * (jnp.array([1, 2, 3, 4, 5]),)
    val_data = 2 * (jnp.array([6, 7, 8]),)

    class Model(eqx.Module):
        def __call__(self, x):
            return 1.1 * x

    model = Model()
    flat_model, treedef_model = jax.tree_util.tree_flatten(model)
    # <<<

    cbargs = CallbackArgs(get_loss, treedef_model, data, val_data)
    dhc = DefaultHistoryCallback(log_every=2)

    # First update
    cbargs.update(flat_model, 1)
    dhc(cbargs)
    assert len(dhc.loss) == 0
    assert len(dhc.val_loss) == 0

    # Second update
    cbargs.update(flat_model, 2)
    dhc(cbargs)
    assert len(dhc.loss) == 1
    assert len(dhc.val_loss) == 1
