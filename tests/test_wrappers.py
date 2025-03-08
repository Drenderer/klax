import klax
import jax.random as jrandom
import jax.numpy as jnp
from jaxtyping import Array
import paramax as px
import pytest


def test_non_negative(getkey):
    # Negative array input
    parameter = -jrandom.uniform(getkey(), (10,))
    non_neg = klax.wrappers.NonNegative(parameter)
    assert jnp.all(px.unwrap(non_neg) == 0) 

    # Positive array input
    parameter = jrandom.uniform(getkey(), (10,))
    non_neg = klax.wrappers.NonNegative(parameter)
    assert jnp.all(px.unwrap(non_neg) == parameter)

    # Array output type
    parameter = -jrandom.uniform(getkey(), (10,))
    non_neg = klax.wrappers.NonNegative(parameter)
    assert isinstance(px.unwrap(non_neg), Array)

    parameter = -jrandom.uniform(getkey(), (10,))
    parameter = px.Parameterize(lambda x: x, parameter)
    non_neg = klax.wrappers.NonNegative(parameter)
    assert isinstance(px.unwrap(non_neg), Array)

