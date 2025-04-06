import sys

import jax
import jax.numpy as jnp
import jax.random as jrandom

from klax.callbacks import CallbackArgs, HistoryCallback


def test_callbackargs(getkey, getmodel, getloss):
    # Test lazy evaluation of loss function
    count = 0

    def get_loss(model, data):
        nonlocal count
        count += 1
        x, y = data
        y_pred = jax.vmap(model)(x)
        return jnp.mean(jnp.square(y - y_pred))

    x = jrandom.uniform(getkey(), (10,))
    x_val = jrandom.uniform(getkey(), (10,))

    model = getmodel()
    flat_model, treedef_model = jax.tree_util.tree_flatten(model)

    cbargs = CallbackArgs(get_loss, treedef_model, (x, x), (x_val, x_val))
    cbargs.update(flat_model, 1)

    assert count == 0
    loss_1 = cbargs.loss
    loss_2 = cbargs.loss
    assert loss_1 == 0
    assert loss_1 == loss_2
    assert count == 1
    _ = cbargs.val_loss
    assert count == 2
    cbargs.update(flat_model, 2)
    _ = cbargs.loss
    assert count == 3

    # Test data reference count
    x = jrandom.uniform(getkey(), (10,))
    data = (x, x)
    model = getmodel()
    flat_model, treedef_model = jax.tree_util.tree_flatten(model)

    assert sys.getrefcount(data) == 2
    cbargs = CallbackArgs(getloss, treedef_model, data)
    assert sys.getrefcount(data) == 3

    # Test update time
    x = jrandom.uniform(getkey(), (10,))
    data = (x, x)
    model = getmodel()
    flat_model, treedef_model = jax.tree_util.tree_flatten(model)
    cbargs = CallbackArgs(getloss, treedef_model, data)
    cbargs.update(flat_model, 0)
    time_on_last_update = cbargs.time_on_last_update
    cbargs.update(flat_model, 0)
    assert cbargs.time_on_last_update >= time_on_last_update


def test_history_callback(getkey, getmodel, getloss):
    x = jrandom.uniform(getkey(), (10,))
    model = getmodel()
    flat_model, treedef_model = jax.tree_util.tree_flatten(model)

    cbargs = CallbackArgs(getloss, treedef_model, (x, x), None)
    history = HistoryCallback(2)

    # On training start update
    cbargs.update(flat_model, 0)
    history.on_training_start(cbargs)

    # First update
    cbargs.update(flat_model, 1)
    history(cbargs)
    assert len(history.loss) == 0
    assert len(history.val_loss) == 0

    # Second update
    cbargs.update(flat_model, 2)
    history(cbargs)
    assert len(history.loss) == 1
    assert len(history.val_loss) == 1

    # On training end update
    cbargs.update(flat_model, -1)
    history.on_training_end(cbargs)

    assert history.training_time > 0.0
