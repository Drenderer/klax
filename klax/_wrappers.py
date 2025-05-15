"""This module implements parameter constraints based on paramax."""

from __future__ import annotations

import jax
from jaxtyping import Array
from paramax import AbstractUnwrappable


# Define a ParameterWrapper as a AbstractUnwrappable wrapping an array or
# an arbitrary depth composition of AbstractUnwrappables around an array.
from typing import Union

ParameterWrapper = AbstractUnwrappable[Union[Array, "ParameterWrapper"]]


class NonNegative(ParameterWrapper):
    """Applies a non-negative constraint by passing the weight
    trough softplus.

    Args:
        parameter: The parameter that is to be made non-negative. It can either
            be a `jax.Array` or a `paramax.AbstractUnwrappable`that is wrapped
            around a `jax.Array`.
    """

    parameter: Array

    def unwrap(self) -> Array:
        return jax.nn.softplus(self.parameter)
