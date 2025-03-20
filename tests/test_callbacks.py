import equinox as eqx
import jax
import jax.numpy as jnp

from klax.callbacks import CallbackArgs

def test_callbackargs():

    computation_counter = 0
    def get_loss(model, x, y):
        nonlocal computation_counter
        computation_counter += 1
        y_pred = jax.vmap(model)(x)
        return jnp.mean(jnp.square(y - y_pred))
    
    data = 2*(jnp.array([1,2,3,4,5]),)
    val_data = 2*(jnp.array([6,7,8]),)

    class Model(eqx.Module):
        def __call__(self, x):
            return 1.1*x
        
    model = Model()
    flat_model, treedef_model = jax.tree_util.tree_flatten(model)

    cbargs = CallbackArgs(get_loss, data, val_data, treedef_model)
    cbargs.update(flat_model, step=1)


    assert computation_counter == 0
    first_computation  = cbargs.loss
    second_computation = cbargs.loss
    assert first_computation == second_computation
    assert computation_counter == 1
    _ = cbargs.val_loss
    assert computation_counter == 2
    cbargs.update(flat_model, step=2)
    _ = cbargs.loss
    assert computation_counter == 3


