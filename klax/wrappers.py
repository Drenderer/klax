"""
This module implements parameter constraints based on paramax.
"""

from abc import abstractmethod

import jax.numpy as jnp
from jaxtyping import Array
import paramax as px


class ParameterWrapper(px.AbstractUnwrappable[Array]):
    """An abstract class representing parameter wrappers.
    
    ParameterWrappers replace PyTree leafs, applying custom heaviour upon
    unwrapping"""
    def __init__(self, parameter: Array | px.AbstractUnwrappable[Array]):
        raise NotImplementedError("To be implemented by derived classes")

    @abstractmethod
    def unwrap(self) -> Array:
        pass


class NonNegative(ParameterWrapper):
    """Applies a non-negative constraint.
    
    **Arguments**:

     - parameter: The parameter that is to be made non-negative. It can either be
        a `jax.Array` or a `paramax.AbstractUnwrappable`that is wrapped around a
        `jax.Array`.
    """

    parameter: Array

    def __init__(self, parameter: Array | px.AbstractUnwrappable[Array]):
        # Ensure that the parameter fulfills the constraint initially
        self.parameter = self._non_neg(px.unwrap(parameter))
    
    def _non_neg(self, x: Array) -> Array:
        return jnp.maximum(x, 0)

    def unwrap(self) -> Array:
        return self._non_neg(self.parameter)
    