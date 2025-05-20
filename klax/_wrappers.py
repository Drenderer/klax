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

"""Parameter constraints based on paramax."""

from __future__ import annotations
from abc import abstractmethod

import jax.numpy as jnp
from jaxtyping import Array
import paramax as px


# Alias to px.unwrap
unwrap = px.unwrap

class ParameterWrapper(px.AbstractUnwrappable[Array]):
    """An abstract class representing parameter wrappers.

    ParameterWrappers replace PyTree leafs, applying custom behaviour upon
    unwrapping"""

    def __init__(self, parameter: Array | px.AbstractUnwrappable[Array]):
        raise NotImplementedError("To be implemented by derived classes")

    @abstractmethod
    def unwrap(self) -> Array:
        pass


class NonNegative(ParameterWrapper):
    """Applies a non-negative constraint.

    Args:
        parameter: The parameter that is to be made non-negative. It can either
            be a `jax.Array` or a `paramax.AbstractUnwrappable`that is wrapped
            around a `jax.Array`.
    """

    parameter: Array

    def __init__(self, parameter: Array | px.AbstractUnwrappable[Array]):
        # Ensure that the parameter fulfills the constraint initially
        self.parameter = self._non_neg(px.unwrap(parameter))

    def _non_neg(self, x: Array) -> Array:
        return jnp.maximum(x, 0)

    def unwrap(self) -> Array:
        return self._non_neg(self.parameter)
