
from jax.nn.initializers import Initializer
from jaxtyping import PRNGKeyArray
from jax import numpy as jnp

def transposed_initialize(initializer:Initializer, key:PRNGKeyArray, shape:tuple, dtype=jnp.float_):
    """
    Calls initializer with the transposed shape and transposes its output.
    Per default, `jax.nn.initilizers` assume that the last axes of a weight matrix
    corresponds to the output axes, while the second last is the input axis. This is 
    likely because JAX assumes per default that the inputs will be multiplied from the left with 
    the weight matrix. However, since we use vmap and don't have to handle batches here, it 
    is possible to multiply the inputs from the right, which aligns with how the math is 
    usually written. In code the klax (and equinox.nn.Linear) modules use the @ operator for this.
    

    Args:
        initializer: `jax.nn.initializers.Initializer`
        key: A `jax.random.PRNGKey`.
        shape: Shape of the array with `(..., out_size, in_size)`.
        dtype: Data type of the initialization. Defaults to jnp.float_.

    Returns:
        Initialized array.
    """
    shape = shape[:-2] + (shape[-1], shape[-2]) # Transpose shape
    return jnp.matrix_transpose(initializer(key, shape, dtype)) # Call and transpose back