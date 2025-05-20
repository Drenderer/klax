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

import equinox as eqx
from klax import (
    apply,
    NonNegative,
    non_trainable,
    Parameterize,
    unwrap
)
import jax.random as jrandom
import jax.numpy as jnp


def test_nested_unwrap():
    param = Parameterize(
        jnp.square,
        Parameterize(jnp.square, Parameterize(jnp.square, 2)),
    )
    assert unwrap(param) == jnp.square(jnp.square(jnp.square(2)))


def test_parameterize():
    diag = Parameterize(jnp.diag, jnp.ones(3))
    assert jnp.allclose(jnp.eye(3), unwrap(diag))


def test_non_trainable():
    model = non_trainable((jnp.ones(3), 1))
    def loss(model):
        model = unwrap(model)
        return model[0].sum()

    grad = eqx.filter_grad(loss)(model)[0].tree
    assert grad.shape == (3,)
    assert jnp.all(grad == 0.0)


#TODO: Paramax implements some more tests 


def test_non_negative(getkey):
    # Negative array input
    parameter = -jrandom.uniform(getkey(), (10,))
    non_neg = NonNegative(parameter)
    assert jnp.all(unwrap(non_neg) == 0)
    assert jnp.all(apply(non_neg).parameter == 0)

    # Positive array input
    parameter = jrandom.uniform(getkey(), (10,))
    non_neg = NonNegative(parameter)
    assert jnp.all(unwrap(non_neg) == parameter)
    assert jnp.all(apply(non_neg).parameter == parameter)
