import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array

import klax

# ===---------------------------------------------------------------------=== #
# klax.count_parameters
# ===---------------------------------------------------------------------=== #


class TestParameterCount:
    @staticmethod
    def test_pytree_with_jax_leaves():
        tree = (jnp.ones((3, 3)), jnp.zeros((2, 3)))
        assert klax.count_parameters(tree) == 15

    @staticmethod
    def test_pytree_with_numpy_leaves():
        tree = (np.ones((3, 3)), np.zeros((2, 3)))
        assert klax.count_parameters(tree) == 15

    @staticmethod
    def test_on_equinox_module():
        class DummyModel(eqx.Module):
            no_parameter: int
            weight: tuple[Array, Array]
            non_trainable: klax.NonTrainable

            def __init__(
                self,
            ):
                self.no_parameter = 1
                self.weight = (jnp.ones((3, 3)), jnp.zeros((3, 3)))
                self.non_trainable = klax.NonTrainable(jnp.ones((2, 2)))

        dummy_model = DummyModel()
        assert klax.count_parameters(dummy_model) == 18
