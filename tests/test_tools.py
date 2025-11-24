import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array

from klax import NonTrainable, parameter_count


def test_parameter_count():
    tree = (jnp.ones((3, 3)), jnp.zeros((2, 3)))
    assert parameter_count(tree) == 15

    class DummyModel(eqx.Module):
        no_parameter: int
        weight: tuple[Array, Array]
        non_trainable: NonTrainable

        def __init__(
            self,
        ):
            self.no_parameter = 1
            self.weight = (jnp.ones((3, 3)), jnp.zeros((3, 3)))
            self.non_trainable = NonTrainable(jnp.ones((2, 2)))

    dummy_model = DummyModel()
    assert parameter_count(dummy_model) == 18
