"""
This module implements parameter constraints based on paramax.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
import paramax


class NonNegative(paramax.AbstractUnwrappable):
    """Applies non-negative constraint to each element of a weight."""
    parameter: Array = eqx.field(converter=lambda x: jnp.maximum(x, 0.))    # Ensure parameters fulfill the constraint initially

    def unwrap(self):
        return jnp.maximum(self.parameter, 0.)
    