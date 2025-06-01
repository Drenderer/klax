# Copyright 2025 The Klax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
def getzerowrap():
    import klax

    class ZeroWrapper(klax.Unwrappable[Array]):
        """A dummy wrapper that sets all parameters to zero."""

        parameter: Array

        def unwrap(self) -> Array:
            return jnp.zeros_like(self.parameter)

    return ZeroWrapper


@pytest.fixture
def getarraywrap():
    import equinox as eqx
    import klax
    from typing import Self

    class Wrapper(klax.Constraint):
        """A dummy wrapper that multiplies the parameter by 2 using the apply functionality."""

        parameter: Array

        def unwrap(self) -> Array:
            return self.parameter

        def apply(self) -> Self:
            return eqx.tree_at(
                lambda x: x.parameter,
                self,
                replace_fn=lambda x: 2 * x,
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
