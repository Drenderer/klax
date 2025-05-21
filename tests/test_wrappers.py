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

import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Array
import paramax as px

from klax import NonNegative, Symmetric, SkewSymmetric, unwrap

def test_non_negative(getkey):
    # Negative array input
    parameter = -jr.uniform(getkey(), (10,))
    non_neg = NonNegative(parameter)
    assert jnp.all(unwrap(non_neg) == 0)

    # Positive array input
    parameter = jr.uniform(getkey(), (10,))
    non_neg = NonNegative(parameter)
    assert jnp.all(unwrap(non_neg) == parameter)

    # Array output type
    parameter = -jr.uniform(getkey(), (10,))
    non_neg = NonNegative(parameter)
    assert isinstance(unwrap(non_neg), Array)

    parameter = -jr.uniform(getkey(), (10,))
    parameter = px.Parameterize(lambda x: x, parameter)
    non_neg = NonNegative(parameter)
    assert isinstance(unwrap(non_neg), Array)

def test_symmetric(getkey):
    # Constraint
    parameter = jr.normal(getkey(), (3,10,3,3))
    symmetric = Symmetric(parameter)
    _symmetric = unwrap(symmetric)
    assert _symmetric.shape == parameter.shape
    assert jnp.array_equal(_symmetric, jnp.transpose(_symmetric, axes=(0,1,3,2)))

def test_skewsymmetric(getkey):
    # Constraint
    parameter = jr.normal(getkey(), (3,10,3,3))
    symmetric = SkewSymmetric(parameter)
    _symmetric = unwrap(symmetric)
    assert _symmetric.shape == parameter.shape
    assert jnp.array_equal(_symmetric, -jnp.transpose(_symmetric, axes=(0,1,3,2)))