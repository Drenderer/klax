from jax.nn.initializers import uniform, he_normal
import jax.numpy as jnp
import jax.random as jrandom
import klax
import pytest


def test_linear(getkey):
    # Zero input shape
    linear = klax.nn.Linear(0, 4, uniform(), key=getkey())
    x = jrandom.normal(getkey(), (0,))
    assert linear(x).shape == (4,)

    # Zero output shape
    linear = klax.nn.Linear(4, 0, uniform(), key=getkey())
    x = jrandom.normal(getkey(), (4,))
    assert linear(x).shape == (0,)

    # Positional arguments
    linear = klax.nn.Linear(3, 4, uniform(), key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    # Some keyword arguments
    linear = klax.nn.Linear(
        3,
        out_features=4,
        weight_init=uniform(),
        key=getkey()
    )
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    # All keyword arguments
    linear = klax.nn.Linear(
        in_features=3,
        out_features=4,
        weight_init=uniform(),
        key=getkey()
    )
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    linear = klax.nn.Linear("scalar", 2, uniform(), key=getkey())
    x = jrandom.normal(getkey(), ())
    assert linear(x).shape == (2,)

    linear = klax.nn.Linear(2, "scalar", uniform(), key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert linear(x).shape == ()

    linear = klax.nn.Linear(
        2,
        "scalar",
        uniform(),
        key=getkey(),
        dtype=jnp.float16
    )
    x = jrandom.normal(getkey(), (2,), dtype=jnp.float16)
    assert linear(x).dtype == jnp.float16

    linear = klax.nn.Linear(
        2,
        "scalar",
        he_normal(), # since uniform does not acces complex numbers
        key=getkey(),
        dtype=jnp.complex64
    )
    x = jrandom.normal(getkey(), (2,), dtype=jnp.complex64)
    assert linear(x).dtype == jnp.complex64