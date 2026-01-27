import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array

from klax import NonTrainable, parameter_count


class TestParameterCount:
    def test_on_simple_pytree(self):
        tree = (jnp.ones((3, 3)), jnp.zeros((2, 3)))
        assert parameter_count(tree) == 15

    def test_on_equinox_module(self):
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
