"""
This module implements a basic training loop.
"""

#TODO: Rewrite docstrings in correct format.
#TODO: Check type annotations.
#TODO: Add features: Implement history object to better handle training history and potentially metrics
#                    Improve handling of the progress printing in fit


from datetime import timedelta
import time
from typing import Callable, Generator, Optional, Tuple, TypeVar, Any, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree
import numpy as np
import optax
import paramax


def dataloader(
    data: Sequence[ArrayLike],
    batch_size: int = 32,
    batch_mask: Sequence[bool] | None = None,
    *,
    key:PRNGKeyArray,
) -> Generator[PyTree, None, None]:
    """Returns a batch `Generator` that yields randomly chosen subsets of data
    without replacement.

    The data may be an any `Sequence` of `ArrayLike`. If `batch_mask` is passed, elements
    without batch dimension can be specified.    

    !!! example

        This is an example for a nested `Sequence`, where the first and the last
        element have batch dimension along the first and the second axes, respectively.
    
        ```python

            x = jnp.array([1., 2.])
            y = jnp.array([[1.], [2.]])
            data = (x, ("unbatched", y))
            batch_mask = (True, (False, True))
            iter_data = dataloader(
                data,
                32,
                batch_mask,
                key=jax.random.PRNGKey(0)
            )
        ```

    **Arguments:**

     - `data`: The `Sequence` of `ArrayLike` data that shall be batched. 
     - `batch_size`: The number of examples in a batch.
     - `batch_mask`: The sequence denoting, which elements of `data` have batch
        dimension. `batch_mask` must have the same `Sequence`structure as `data`, where
         the elements are replaced with values of type `bool`. `True` indicates
         that the corresponding element in `data` has batch dimension. If `False`, the
         corresponding element will be returned unchanged by the `Generator`.
         (Defaults to `None`, meaning all elements of `data` have batch dimension.)
     - `key`: A `jax.random.PRNGKey` used to provide randomness for batch generation.
         (Keyword only argument.)

    Note that the batch dimension for all batched elements must correspond to the first
    array dimension.

    **Returns:**

    A `Generator` that yields a random batch of data every time is is called.

    **Yields:**
    
    A `Sequence[ArrayLike]` with the same structure as data. Where the non-masked
    leafs have a size of `batch_size`.
    """

    # Generate an all true batch mask if batch_mask = None was passed
    if batch_mask is None:
        batch_mask = jax.tree.map(lambda x: x is not None, data)

    # Split the PyTree according to the batch mask
    batched_data, unbatched_data = eqx.partition(data, batch_mask)

    # Check that all batched data has the same batch dimension along their
    # respective batch axes
    data_leafs = jax.tree.leaves(batched_data)
    if len(data_leafs) == 0:
        raise ValueError("At least one element must have a batch dimension.")
    dataset_size = batched_data[0].shape[0]
    if not all(arr.shape[0] == dataset_size for arr in data_leafs[1:]):
        raise ValueError("All batched arrays must have the same batch dimension.")

    # Convert to Numpy arrays. Numpy's slicing is much faster than Jax's, so for
    # fast model training steps this actually makes a huge difference!
    batched_data = jax.tree.map(lambda x: np.array(x), batched_data) 

    # Reduce batch size if the dataset has less examples than batch size
    batch_size = min(batch_size, dataset_size)

    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        print(perm)
        (key,) = jr.split(key, 1) # Update key
        start, end = 0, batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            bs = jax.tree.map(lambda x: x[batch_perm], batched_data)
            yield eqx.combine(bs, unbatched_data)
            start = end
            end = start + batch_size


# def fit(model: T,
#         x: PyTree[ArrayLike|None],
#         y: PyTree[ArrayLike|None],
#         *,
#         validation_data: Optional[tuple[PyTree[ArrayLike|None], PyTree[ArrayLike|None]]] = None,
#         batch_size: int = 32,
#         batch_mask: Optional[PyTree[bool]] = None,
#         steps: int = 1000,
#         log_loss_every: int = 100,
#         loss_fn: Callable[..., float] = _mse,
#         optimizer: optax.GradientTransformation = optax.adam(1e-3),
#         callback: Optional[Callable] = None,
#         key: PRNGKeyArray,
#         ) -> tuple[T, dict]:
# 
#     """Trains a model using an optimizer from optax.
# 
#     Args:
#         model (Module): The model instance which should be trained. It may contain instances of Constraint classes. Should be a subclass of eqx.Module.
#         x (Array|tuple[Array, ...]): Input data. It could be a jax.numpy.array or a pytree of jax.numpy.array instances.
#         y (Array|tuple[Array, ...]): Target data. It could be a jax.numpy.array or a pytree of jax.numpy.array instances.
#         validation_data (tuple, optional): A tuple of inputs and targets used for validation during training. Defaults to None.
#         batch_size (int, optional): Number of samples per gradient update. Defaults to 32.
#         batch_mask (PyTree, optional): A tuple-pytree with booleans as leafs and the same structure as the pytree ```(x, y)```. The booleans indicate if the corresponding array in the data pytree has a batch dimension. If False, then the corresponding data array will be passed unchanged every time. None indicates that all arrays have a batch dimension. Defaults to None.
#         steps (int, optional): Number of gradient updates to apply. Defaults to 1000.
#         log_loss_every (int, optional): Idicates how many steps need to be taken in order to conduct a new loss evaluation. A loss evaluation consists of calculating the trainin and validation losses *over the etire datasets* and storing them in the history dictionary. Defaults to 10.
#         loss_fn (Callable): A function with call signature ```(prediction, target, model) -> float``` that computes the loss. Defaults to mse.
#         optimizer (optax.GradientTransformation): Any optax gradient transform to calculate the updates for the model. Defaults to optax.adam(1e-3).
#         callback (Callable, optional): A function that is called after every gradient update. The call signature is ```(model, step) -> None```.
#         key (PRNGKeyArray): A PRNGKey to randomize the individual batches.
#         
#     Returns:
#         model, history (tuple[eqx.Module, dict]): Returns a tuple of the trained model and a history dictionary containing the loss history.
#     """
# 
#     # Determine the batch dimension for each leaf in the input pytree according to the batch mask
#     if batch_mask is None:
#         model_in_axes = 0
#     else:
#         model_in_axes = (jax.tree.map(lambda x: 0 if x else None, batch_mask[0]),)
# 
#     # Define a function to calculate the loss. This is jit compiled to speed up the loss evaluation for the loss history.
#     @eqx.filter_jit
#     def get_loss(model, x, y):
#         model = paramax.unwrap(model)
#         y_pred = jax.vmap(model, in_axes=model_in_axes)(x)
#         return loss_fn(y_pred, y, model)
#     
#     grad_loss = eqx.filter_grad(get_loss) # Get the gradient function
# 
#     @eqx.filter_jit
#     def make_step(x, y, flat_model, optimizer, flat_opt_state):
#         # Use the unflatten trick to speed up training, see https://docs.kidger.site/equinox/tricks/
#         model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
#         opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, flat_opt_state)
# 
#         # Compute and apply the parameter updates
#         grads = grad_loss(model, x, y)
#         updates, opt_state = optimizer.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
#         model = eqx.apply_updates(model, updates)
# 
#         flat_model = jax.tree_util.tree_leaves(model)
#         flat_opt_state = jax.tree_util.tree_leaves(opt_state)
# 
#         return flat_model, flat_opt_state
# 
#     # Initialize the history dict
#     history = {'log_loss_every': log_loss_every,
#                'loss': [],}
#     vx, vy = None, None
#     if validation_data is not None:
#         vx, vy = validation_data
#         history['val_loss'] = []
# 
#     val_loss = None
# 
#     # Initialize the optimizer and 'tell it' to optimize with respect to all inexact arrays in the model
#     opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
# 
#     # Use the unflatten trick to speed up training, see https://docs.kidger.site/equinox/tricks/
#     flat_model, treedef_model = jax.tree_util.tree_flatten(model)
#     flat_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state)
# 
#     # Loop over all training steps
#     start_time = time.time()
#     for step, (xi, yi) in zip(range(1, steps+1), 
#                               _dataloader((x, y), batch_size, batch_mask=batch_mask, key=key)):
#         flat_model, flat_opt_state = make_step(xi, yi, flat_model, optimizer, flat_opt_state)   # Make the step
# 
#         if callback is not None:
#             model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
#             callback(model, step)
# 
#         # Log the losses
#         if (step % log_loss_every) == 0 or step == steps - 1:
#             model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
#             train_loss = get_loss(model, x, y)
#             history['loss'].append(train_loss)
#             if validation_data is not None:
#                 val_loss = get_loss(model, vx, vy)
#                 history['val_loss'].append(val_loss)
#                 print(f"Step: {step}, Loss: {train_loss:.3e}, Validation loss: {val_loss:.3e}")
#             else:
#                 print(f"Step: {step}, Loss: {train_loss:.3e}")
# 
#     model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
# 
#     training_time = time.time() - start_time
#     print(f'Training took: {timedelta(seconds=training_time)}')
# 
#     history['training_time'] = training_time
#     history = {k: np.array(v) for k,v in history.items()}
# 
#     return model, history
# 
# 
# 
# 
# 