import jax
import pytest

from klax import canonicalize_initializer, hoedt_bias, hoedt_normal


@pytest.mark.parametrize(
    "init",
    [
        pytest.param(jax.nn.initializers.constant(1.0), id="constant"),
        pytest.param(
            jax.nn.initializers.delta_orthogonal(), id="delta_orthogonal"
        ),
        pytest.param(jax.nn.initializers.glorot_normal(), id="glorot_normal"),
        pytest.param(
            jax.nn.initializers.glorot_uniform(), id="glorot_uniform"
        ),
        pytest.param(jax.nn.initializers.he_normal(), id="he_normal"),
        pytest.param(jax.nn.initializers.he_uniform(), id="he_uniform"),
        pytest.param(
            jax.nn.initializers.kaiming_normal(), id="kaiming_normal"
        ),
        pytest.param(
            jax.nn.initializers.kaiming_uniform(), id="kaiming_uniform"
        ),
        pytest.param(jax.nn.initializers.lecun_normal(), id="lecun_normal"),
        pytest.param(jax.nn.initializers.lecun_uniform(), id="lecun_uniform"),
        pytest.param(jax.nn.initializers.normal(), id="normal"),
        pytest.param(jax.nn.initializers.ones, id="ones"),
        pytest.param(jax.nn.initializers.orthogonal(), id="orthogonal"),
        pytest.param(
            jax.nn.initializers.truncated_normal(), id="truncated_normal"
        ),
        pytest.param(jax.nn.initializers.uniform(), id="uniform"),
        pytest.param(
            jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal"),
            id="variance_scaling",
        ),
        pytest.param(jax.nn.initializers.xavier_normal(), id="xavier_normal"),
        pytest.param(
            jax.nn.initializers.xavier_uniform(), id="xavier_uniform"
        ),
        pytest.param(jax.nn.initializers.zeros, id="zeros"),
        pytest.param(hoedt_bias(), id="hoedt_bias"),
        pytest.param(hoedt_normal(), id="hoedt_normal"),
    ],
)
def test_canonicalize_initializer(getkey, init):
    """Test all klax and jax initializers."""
    init = canonicalize_initializer(init)
    init(getkey(), (2, 1, 3), 2)
