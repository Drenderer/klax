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

import klax
import jax.random as jrandom
import jax.numpy as jnp
from jaxtyping import Array
import paramax as px

def test_non_negative(getkey):
    # Negative array input
    parameter = -jrandom.uniform(getkey(), (10,))
    non_neg = klax.NonNegative(parameter)
    assert jnp.all(px.unwrap(non_neg) == 0)

    # Positive array input
    parameter = jrandom.uniform(getkey(), (10,))
    non_neg = klax.NonNegative(parameter)
    assert jnp.all(px.unwrap(non_neg) == parameter)

    # Array output type
    parameter = -jrandom.uniform(getkey(), (10,))
    non_neg = klax.NonNegative(parameter)
    assert isinstance(px.unwrap(non_neg), Array)

    parameter = -jrandom.uniform(getkey(), (10,))
    parameter = px.Parameterize(lambda x: x, parameter)
    non_neg = klax.NonNegative(parameter)
    assert isinstance(px.unwrap(non_neg), Array)
