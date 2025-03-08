"""
This module implements a basic training loop.
"""

from typing import Generator, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree
import numpy as np


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

        This is an example for a nested `Sequence`, where the elements x and y
        have batch dimension along the first axis.
    
        ```python

            x = jnp.array([1., 2.])
            y = jnp.array([[1.], [2.]])
            data = (x, (1.0, y))
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
    print(batch_mask)

    # Split the PyTree according to the batch mask
    batched_data, unbatched_data = eqx.partition(data, batch_mask)

    # Check that all batched data has the same batch dimension along their
    # respective batch axes
    batched_leafs = jax.tree.leaves(batched_data)
    if len(batched_leafs) == 0:
        raise ValueError("At least one element must have a batch dimension.")
    dataset_size = batched_leafs[0].shape[0]
    for arr in batched_leafs[1:]:
        print(arr.shape[0])
    if not all(arr.shape[0] == dataset_size for arr in batched_leafs[1:]):
        raise ValueError("All batched arrays must have the same batch dimension.")

    # Convert to Numpy arrays. Numpy's slicing is much faster than Jax's, so for
    # fast model training steps this actually makes a huge difference!
    batched_data = jax.tree.map(lambda x: np.array(x), batched_data) 

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
