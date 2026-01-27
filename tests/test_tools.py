import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array

from klax import NonTrainable, count_parameters


class TestParameterCount:
    def test_pytree_with_jax_leaves(self):
        tree = (jnp.ones((3, 3)), jnp.zeros((2, 3)))
        assert count_parameters(tree) == 15

    def test_pytree_with_numpy_leaves(self):
        tree = (np.ones((3, 3)), np.zeros((2, 3)))
        assert count_parameters(tree) == 15

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
        assert count_parameters(dummy_model) == 18
