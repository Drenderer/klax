import jax
import pytest

from klax import canonicalize_initializer, hoedt_bias, hoedt_normal


@pytest.mark.parametrize(
    "init",
    [
        jax.nn.initializers.constant(1.0),
        jax.nn.initializers.delta_orthogonal(),
        jax.nn.initializers.glorot_normal(),
        jax.nn.initializers.glorot_uniform(),
        jax.nn.initializers.he_normal(),
        jax.nn.initializers.he_uniform(),
        jax.nn.initializers.kaiming_normal(),
        jax.nn.initializers.kaiming_uniform(),
        jax.nn.initializers.lecun_normal(),
        jax.nn.initializers.lecun_uniform(),
        jax.nn.initializers.normal(),
        jax.nn.initializers.ones,
        jax.nn.initializers.orthogonal(),
        jax.nn.initializers.truncated_normal(),
        jax.nn.initializers.uniform(),
        jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal"),
        jax.nn.initializers.xavier_normal(),
        jax.nn.initializers.xavier_uniform(),
        jax.nn.initializers.zeros,
        hoedt_bias(),
        hoedt_normal(),
    ],
)
def test_canonicalize_initializer(getkey, init):
    # Test all klax and JAX initializers
    init = canonicalize_initializer(init)
    init(getkey(), (2, 1, 3), 2)
