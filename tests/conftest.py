import typing

import jax
import jax.numpy as jnp
from jaxtyping import Array
import pytest


typing.TESTING = True  # pyright: ignore


# jax.config.update("jax_numpy_dtype_promotion", "strict")  # Causes issues because implicit data type promotion is used in klax.fit since the default batch_data works with numpy arrays
jax.config.update("jax_numpy_rank_promotion", "raise")


@pytest.fixture
def getkey():
    # Delayed import so that jaxtyping can transform the AST of Equinox before it is
    # imported, but conftest.py is ran before then.
    import equinox.internal as eqxi

    return eqxi.GetKey()


@pytest.fixture
def getwrap():
    import equinox as eqx
    import klax
    from typing import Self

    # Implementation of a dummy wrapper that sets all parameters to zero.
    class Wrapper(klax.ArrayWrapper):
        parameter: Array

        def __init__(self, parameter: Array):
            self.parameter = jnp.zeros_like(parameter)

        def unwrap(self) -> Array:
            return jnp.zeros_like(self.parameter)

        def apply(self) -> Self:
            return eqx.tree_at(
                lambda x: x.parameter,
                self,
                replace_fn=lambda x: jnp.zeros_like(x),
            )

    return Wrapper


@pytest.fixture
def getmodel():
    import equinox as eqx

    class Model(eqx.Module):
        def __call__(self, x):
            return x

    return Model


@pytest.fixture
def getloss():
    def init(model, data):
        x, y = data
        y_pred = jax.vmap(model)(x)
        return jnp.mean(jnp.square(y - y_pred))

    return init
