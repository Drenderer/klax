import jax
import jax.numpy as jnp


def default_floating_dtype():
    """This function was copied from equinox._misc"""
    if jax.config.jax_enable_x64:
        return jnp.float64
    else:
        return jnp.float32