import klax
import jax.random as jrandom
import jax.numpy as jnp
from jaxtyping import Array
import paramax as px


def test_updatables():
    class UpWrap(klax.AbstractUpdatable[Array]):
        def update(self):
            return self.parameter + 1.0

    # Basic Updatable usage
    value = jnp.array([1.0, 2.0, 3.0])
    param = UpWrap(value)
    param_ = klax.update_wrapper(param)
    assert isinstance(param_, UpWrap)
    assert jnp.all(klax.unwrap(param_) == value + 1.0)

    # Nested Updatables
    value = jnp.array([1.0, 2.0, 3.0])
    param = UpWrap(UpWrap(value))
    param_ = klax.update_wrapper(param)
    assert isinstance(param_, UpWrap)
    assert isinstance(param_.parameter, UpWrap)
    assert jnp.all(klax.unwrap(param_) == value + 2.0)


def test_positive_finalizepable(getkey):
    # Negative array input
    parameter = -jrandom.uniform(getkey(), (10,))
    non_neg = klax.Positive(parameter)
    assert isinstance(klax.finalize(non_neg), Array)
    assert jnp.all(klax.finalize(non_neg) > 0)

    # Positive array input
    parameter = jrandom.uniform(getkey(), (10,))
    non_neg = klax.Positive(parameter)
    assert jnp.all(klax.finalize(non_neg) > parameter)

    # Wrapper composition
    parameter = -jrandom.uniform(getkey(), (10,))
    parameter = px.Parameterize(lambda x: x, parameter)
    non_neg = klax.Positive(parameter)
    assert isinstance(klax.finalize(non_neg), Array)


def test_non_negative_updatable(getkey):
    # Negative array input
    parameter = -jrandom.uniform(getkey(), (10,))
    non_neg = klax.NonNegative(parameter)
    non_neg = klax.update_wrapper(non_neg)
    assert isinstance(klax.finalize(non_neg), Array)
    assert jnp.all(klax.finalize(non_neg) >= 0)

    # NonNegative array input
    parameter = jrandom.uniform(getkey(), (10,))
    non_neg = klax.NonNegative(parameter)
    non_neg = klax.update_wrapper(non_neg)
    assert jnp.all(klax.finalize(non_neg) >= parameter)
