import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Array
import paramax as px

from klax.wrappers import NonNegative, Symmetric, SkewSymmetric, unwrap

def test_non_negative(getkey):
    # Negative array input
    parameter = -jr.uniform(getkey(), (10,))
    non_neg = NonNegative(parameter)
    assert jnp.all(px.unwrap(non_neg) == 0)

    # Positive array input
    parameter = jr.uniform(getkey(), (10,))
    non_neg = NonNegative(parameter)
    assert jnp.all(px.unwrap(non_neg) == parameter)

    # Array output type
    parameter = -jr.uniform(getkey(), (10,))
    non_neg = NonNegative(parameter)
    assert isinstance(px.unwrap(non_neg), Array)

    parameter = -jr.uniform(getkey(), (10,))
    parameter = px.Parameterize(lambda x: x, parameter)
    non_neg = NonNegative(parameter)
    assert isinstance(px.unwrap(non_neg), Array)

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