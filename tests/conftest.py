import typing

import jax
import jax.numpy as jnp
from jaxtyping import Array
import paramax as px
import pytest


typing.TESTING = True  # pyright: ignore


# jax.config.update("jax_numpy_dtype_promotion", "strict")  # Causes issues because implicit data type promotion is used in klax.fit since the default dataloader works with numpy arrays
jax.config.update("jax_numpy_rank_promotion", "raise")


@pytest.fixture
def getkey():
    # Delayed import so that jaxtyping can transform the AST of Equinox before it is
    # imported, but conftest.py is ran before then.
    import equinox.internal as eqxi

    return eqxi.GetKey()


@pytest.fixture
def getwrap():
    import klax

    # Implementation of a dummy wrapper that sets all parameters to zero.
    class Wrapper(klax.wrappers.ParameterWrapper):
        parameter: Array

        def __init__(self, parameter: Array | px.AbstractUnwrappable[Array]):
            self.parameter = jnp.zeros_like(px.unwrap(parameter))
        
        def unwrap(self) -> Array:
            return jnp.zeros_like(self.parameter)
    
    return Wrapper
