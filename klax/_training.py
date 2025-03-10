"""
This module implements a basic training loop.
"""

from collections.abc import Callable
import datetime
import time
from typing import Any, Generator, Sequence, Tuple, TypeVar, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree
import numpy as np
from numpy.typing import NDArray
import optax
import paramax as px


T = TypeVar("T", bound=eqx.Module)
type _Data = PyTree[ArrayLike | None]
type _BatchGenerator = Generator[_Data, None, None]
type _DataLoader = Callable[[_Data, int, PyTree[bool] | None], _BatchGenerator]
type _LossFunction = Callable[[T, _Data, _Data, int | None | Sequence[Any]], float]


def mse(
    model: T,
    x: _Data,
    y: _Data,
    in_axes: int | None | Sequence[Any] = 0
):
    y_pred = jax.vmap(model, in_axes=in_axes)(x)
    return jnp.mean((y_pred - y) ** 2)


def dataloader(
    data: _Data,
    batch_size: int = 32,
    batch_mask: PyTree[bool] | None = None,
    *,
    key:PRNGKeyArray,
) -> _BatchGenerator:
    """Returns a batch `Generator` that yields randomly chosen subsets of data
    without replacement.

    The data can be any `PyTree` with `ArrayLike` leaves. If `batch_mask` is passed,
    leaves without batch dimension can be specified.    

    !!! example

        This is an example for a nested `PyTree`, where the elements x and y
        have batch dimension along the first axis.
    
        ```python

            x = jnp.array([1., 2.])
            y = jnp.array([[1.], [2.]])
            data = (x, {"a": 1.0, "b": y))
            batch_mask = (True, {"a": False, "b": True})
            iter_data = dataloader(
                data,
                32,
                batch_mask,
                key=jax.random.PRNGKey(0)
            )
        ```

    **Arguments:**

     - `data`: The data that shall be batched. It can be any `PyTree` with `ArrayLike`
         leaves. 
     - `batch_size`: The number of examples in a batch.
     - `batch_mask`: The `PyTree` denoting, which leaves of `data` have batch
        dimension. `batch_mask` must have the same structure as `data`, where
         the leaves are replaced with values of type `bool`. `True` indicates
         that the corresponding leaf in `data` has batch dimension. If `False`, the
         corresponding leaf will be returned unchanged by the `Generator`.
         (Defaults to `None`, meaning all leaves in `data` have batch dimension.)
     - `key`: A `jax.random.PRNGKey` used to provide randomness for batch generation.
         (Keyword only argument.)

    Note that the batch axis for all batched leaves must correspond to the first
    array axis.

    **Returns:**

    A `Generator` that yields a random batch of data every time is is called.

    **Yields:**
    
    A `PyTree[ArrayLike]` with the same structure as `data`. Where all batched
    leaves have `batch_size`.

    Note that if the size of the dataset is smaller than `batch_size`, the obtained
    batches will have dataset size.
    """

    # Generate an all true batch mask if batch_mask = None was passed
    if batch_mask is None:
        batch_mask = jax.tree.map(lambda x: x is not None, data)
    
    # Check that data and batch_mask have the same PyTree structure
    if jax.tree.structure(data) != jax.tree.structure(batch_mask):
        raise ValueError(
            "Arguments data and batch_mask must have equal PyTree structures.")
    
    # Split the PyTree according to the batch mask
    batched_data, unbatched_data = eqx.partition(data, batch_mask)

    # Check that all batched leaves has the same dimension along the first axis
    batched_leafs = jax.tree.leaves(batched_data)
    if len(batched_leafs) == 0:
        raise ValueError("At least one element must have batch dimension.")
    dataset_size = batched_leafs[0].shape[0]
    if not all(arr.shape[0] == dataset_size for arr in batched_leafs[1:]):
        raise ValueError("All batched arrays must have equal batch dimension.")

    # Convert to Numpy arrays. Numpy's slicing is much faster than Jax's, so for
    # fast model training steps this actually makes a huge difference!
    # batched_data = jax.tree.map(lambda x: np.array(x), batched_data) 

    # Reduce batch size if the dataset has less examples than batch size
    batch_size = min(batch_size, dataset_size)

    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1) # Update key
        start, end = 0, batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            bs = jax.tree.map(lambda x: x[batch_perm], batched_data)
            yield eqx.combine(bs, unbatched_data)
            start = end
            end = start + batch_size


def fit(model: T,
        x: _Data,
        y: _Data,
        batch_size: int = 32,
        in_mask: PyTree[bool] | None = None,
        out_mask: PyTree[bool] | None = None,
        *,
        validation_data: Tuple[_Data, _Data] | None = None,
        steps: int = 1000,
        log_every: int = 100,
        loss_fn: _LossFunction = mse,
        optimizer: optax.GradientTransformation = optax.adam(1e-3),
        dataloader: _DataLoader = dataloader,
        key: PRNGKeyArray,
        ) -> Tuple[T, dict]:
    """Trains a model using an optimizer from optax.

    **Arguments**:

     - `model`: The model instance, which should be trained. It must be a subclass of
         `eqx.Module`. The model may contain `paramax.AbstractUnwrappable` wrappers.
     - `x`: The input data. It can be any `PyTree` with `ArrayLike` leaves.
     - `y': The target data. It can be any `PyTree` with `ArrayLike` leaves.
     - `batch_size`: The number of examples in a batch.
     - `in_mask`: The `PyTree` denoting, which leaves of `x` have batch
        dimension. `in_mask` must have the same structure as `x`, where
         the leaves are replaced with values of type `bool`. `True` indicates
         that the corresponding leaf in `x` has batch dimension. If `False`, the
         corresponding leaf will be returned unchanged by the `Generator`.
         (Defaults to `None`, meaning all leaves in `x` have batch dimension.)
     - `out_mask`: The `PyTree` denoting, which leaves of `y` have batch
        dimension. `out_mask` must have the same structure as `y`, where
         the leaves are replaced with values of type `bool`. `True` indicates
         that the corresponding leaf in `y` has batch dimension. If `False`, the
         corresponding leaf will be returned unchanged by the `Generator`.
         (Defaults to `None`, meaning all leaves in `y` have batch dimension.)
     - `validation_data`: A tuple of inputs and targets used for validation during training.
         (Defaults to None. Keyword only argument)
     - `steps`: Number of gradient updates to apply. (Defaults to 1000. Keyword only argument)
     - `log_every`: The number of steps between updates of the loss history. A history update
         consists of calculating the training and validation losses *over the etire datasets*
         and storing them in the history dictionary. (Defaults to 10. Keyword only Argument)
     - `loss_fn`: The loss function with call signature `(model, prediction, target, in_axes) -> float`.
         (Defaults to `mse`.)
     - `optimizer`: The optimizer. Any optax gradient transform to calculate the updates for
        the model. (Defaults to optax.adam(1e-3).)
     - `dataloader`. The data loader that splits inputs and targets into batches.
        (Defaults to `dataloader`)
     - `key`: A `jax.random.PRNGKey` used to provide randomness for batch generation.
         (Keyword only argument.)

    Note that, this function assumes that the batch dimension is always oriented along
    the first axes of any `jax.Array`
        
    **Returns:**

    Returns a tuple of the trained model and a history dictionary containing the loss history.
    """

    # Generate an all true masks if in_axes/out_axes is None
    if in_mask is None:
        in_mask = jax.tree.map(lambda x: x is not None, x)
    if out_mask is None:
        out_mask = jax.tree.map(lambda x: x is not None, y)

    # Check that y and x have the same PyTree structure as in_mask and out_mask
    if jax.tree.structure(x) != jax.tree.structure(in_mask):
        raise ValueError(
            "Arguments x and in_mask must have equal PyTree structure.")
    if jax.tree.structure(y) != jax.tree.structure(out_mask):
        raise ValueError(
            "Arguments y and in_mask must have equal PyTree structure.")

    # Mark the first dimension as batch dimension for all leaves in x that are
    # not masked
    in_axes = jax.tree.map(lambda x: 0 if x else None, in_mask)

    # Define a function to calculate the loss. This is jit compiled to speed up
    # the loss evaluation for the loss history.
    @eqx.filter_jit
    def get_loss(model, x, y):
        model = px.unwrap(model)
        return loss_fn(model, x, y, in_axes=in_axes)

    # Get the gradient function
    grad_loss = eqx.filter_grad(get_loss)

    @eqx.filter_jit
    def make_step(x, y, flat_model, optimizer, flat_opt_state):
        # Use the unflatten trick to speed up training,
        # see https://docs.kidger.site/equinox/tricks/
        model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
        opt_state = jax.tree_util.tree_unflatten(treedef_opt_state, flat_opt_state)

        # Compute and apply the parameter updates
        grads = grad_loss(model, x, y)
        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params=eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)

        flat_model = jax.tree_util.tree_leaves(model)
        flat_opt_state = jax.tree_util.tree_leaves(opt_state)

        return flat_model, flat_opt_state

    # Initialize the history dict
    history = {'steps': [], 'loss': [],}
    vx, vy = None, None
    if validation_data is not None:
        vx, vy = validation_data
        history['val_loss'] = []

    val_loss = None

    # Initialize the optimizer and 'tell it' to optimize with respect to all
    # inexact arrays in the model
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Use the unflatten trick to speed up training,
    # see https://docs.kidger.site/equinox/tricks/
    flat_model, treedef_model = jax.tree_util.tree_flatten(model)
    flat_opt_state, treedef_opt_state = jax.tree_util.tree_flatten(opt_state)

    # Loop over all training steps
    start_time = time.time()
    for step, (xi, yi) in zip(range(1, steps+1), dataloader(
        (x, y),
        batch_size,
        (in_mask, out_mask),
        key=key
    )):
        flat_model, flat_opt_state = make_step(
            xi,
            yi,
            flat_model,
            optimizer,
            flat_opt_state
        )

        # Log the losses
        if (step % log_every) == 0 or step == steps:
            model = jax.tree_util.tree_unflatten(treedef_model, flat_model)
            loss = get_loss(model, x, y)
            history['steps'].append(step)
            history['loss'].append(loss)
            message = f"Step: {step}, Loss: {loss:.3e}"

            if validation_data is not None:
                val_loss = get_loss(model, vx, vy)
                history['val_loss'].append(val_loss)
                message += f", Validation loss: {val_loss:.3e}"

            print(message) 

    model = jax.tree_util.tree_unflatten(treedef_model, flat_model)

    training_time = time.time() - start_time
    print(f'Training took: {datetime.timedelta(seconds=training_time)}')
    history['training_time'] = training_time

    # Convert history to numpy arrays
    history = {k: np.array(v) for k,v in history.items()}

    return model, history