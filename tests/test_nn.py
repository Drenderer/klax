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

    # Scalar shapes
    linear = klax.nn.Linear("scalar", 2, uniform(), key=getkey())
    x = jrandom.normal(getkey(), ())
    assert linear(x).shape == (2,)

    linear = klax.nn.Linear(2, "scalar", uniform(), key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert linear(x).shape == ()

    # Data type
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


def test_fully_linear(getkey):
    # Zero input shape
    fully_linear = klax.nn.FullyLinear(0, 0, 4, uniform(), key=getkey())
    y = jrandom.normal(getkey(), (0,))
    z = jrandom.normal(getkey(), (0,))
    assert fully_linear(y, z).shape == (4,)

    fully_linear = klax.nn.FullyLinear(0, 3, 4, uniform(), key=getkey())
    y = jrandom.normal(getkey(), (0,))
    z = jrandom.normal(getkey(), (3,))
    assert fully_linear(y, z).shape == (4,)

    fully_linear = klax.nn.FullyLinear(3, 0, 4, uniform(), key=getkey())
    y = jrandom.normal(getkey(), (3,))
    z = jrandom.normal(getkey(), (0,))
    assert fully_linear(y, z).shape == (4,)

    # Zero output shape
    fully_linear = klax.nn.FullyLinear(4, 3, 0, uniform(), key=getkey())
    y = jrandom.normal(getkey(), (4,))
    z = jrandom.normal(getkey(), (3,))
    assert fully_linear(y, z).shape == (0,)

    # Positional arguments
    fully_linear = klax.nn.FullyLinear(3, 2, 4, uniform(), key=getkey())
    y = jrandom.normal(getkey(), (3,))
    z = jrandom.normal(getkey(), (2,))
    assert fully_linear(y, z).shape == (4,)

    # Some keyword arguments
    fully_linear = klax.nn.FullyLinear(
        3,
        2,
        out_features=4,
        weight_init=uniform(),
        key=getkey()
    )
    y = jrandom.normal(getkey(), (3,))
    z = jrandom.normal(getkey(), (2,))
    assert fully_linear(y, z).shape == (4,)

    # All keyword arguments
    fully_linear = klax.nn.FullyLinear(
        in_features_y=3,
        in_features_z=2,
        out_features=4,
        weight_init=uniform(),
        key=getkey()
    )
    y = jrandom.normal(getkey(), (3,))
    z = jrandom.normal(getkey(), (2,))
    assert fully_linear(y, z).shape == (4,)

    # Scalar shapes
    fully_linear = klax.nn.FullyLinear("scalar", 2, 3, uniform(), key=getkey())
    y = jrandom.normal(getkey(), ())
    z = jrandom.normal(getkey(), (2,))
    assert fully_linear(y, z).shape == (3,)

    fully_linear = klax.nn.FullyLinear(2, "scalar", 3, uniform(), key=getkey())
    y = jrandom.normal(getkey(), (2,))
    z = jrandom.normal(getkey(), ())
    assert fully_linear(y, z).shape == (3,)

    fully_linear = klax.nn.FullyLinear(
        "scalar",
        "scalar",
        3,
        uniform(),
        key=getkey()
    )
    y = jrandom.normal(getkey(), ())
    z = jrandom.normal(getkey(), ())
    assert fully_linear(y, z).shape == (3,)

    fully_linear = klax.nn.FullyLinear(2, 3, "scalar", uniform(), key=getkey())
    y = jrandom.normal(getkey(), (2,))
    z = jrandom.normal(getkey(), (3,))
    assert fully_linear(y, z).shape == ()

    # Data types
    fully_linear = klax.nn.FullyLinear(
        2,
        3,
        "scalar",
        uniform(),
        key=getkey(),
        dtype=jnp.float16
    )
    y = jrandom.normal(getkey(), (2,), dtype=jnp.float16)
    z = jrandom.normal(getkey(), (3,), dtype=jnp.float16)
    assert fully_linear(y, z).dtype == jnp.float16

    fully_linear = klax.nn.FullyLinear(
        2,
        3,
        "scalar",
        he_normal(),
        key=getkey(),
        dtype=jnp.complex64
    )
    y = jrandom.normal(getkey(), (2,), dtype=jnp.complex64)
    z = jrandom.normal(getkey(), (3,), dtype=jnp.complex64)
    assert fully_linear(y, z).dtype == jnp.complex64
