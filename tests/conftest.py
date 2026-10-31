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

import contextlib
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
