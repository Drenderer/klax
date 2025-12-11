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
import pytest
from jaxtyping import Array

typing.TESTING = True  # pyright: ignore

# Causes issues because implicit data type promotion is used in klax.fit since
# the default batch_data works with numpy arrays
# jax.config.update("jax_numpy_dtype_promotion", "strict")

jax.config.update("jax_numpy_rank_promotion", "raise")


@pytest.fixture
def getkey():
    # Delayed import so that jaxtyping can transform the AST of Equinox before
    # it is imported, but conftest.py is ran before then.
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
    from typing import Self

    import equinox as eqx

    import klax

    class Wrapper(klax.Constraint):
        """A constraint that multiplies the parameter by 2 when applied."""

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
def dummy_model():
    import equinox as eqx

    class Model(eqx.Module):
        def __call__(self, x):
            return x

    return Model()


@pytest.fixture
def dummy_data():
    import jax.random as jr

    x = jr.uniform(jr.key(0), (100, 4))
    y = jnp.abs(x).sum(axis=1, keepdims=True)
    return x, y


@pytest.fixture
def dummy_train_state(dummy_model):
    import equinox as eqx
    import optax

    from klax._trainstate import TrainingState

    opt_state = optax.adam(1e-3).init(
        eqx.filter(dummy_model, eqx.is_inexact_array)
    )
    return TrainingState.create(model=dummy_model, opt_state=opt_state)


@pytest.fixture
def dummy_train_static(getkey, dummy_data):
    import optax

    from klax import TrainingStatic, batch_data, mse

    return TrainingStatic(
        optimizer=optax.adam(1e-3),
        batcher=batch_data(
            dummy_data, batch_size=10, batch_axes=0, key=getkey()
        ),
        batch_axes=0,
        loss=mse,
        steps=1000,
    )


@pytest.fixture
def dummy_metric_fn():
    import equinox as eqx

    def parameter_sum_metric(model):
        return sum(
            x.sum()
            for x in jax.tree_util.tree_leaves(model)
            if eqx.is_array(x)
        )

    return parameter_sum_metric
