import equinox as eqx
import jax
from jaxtyping import PyTree

from ._wrappers import NonTrainable


def parameter_count(model: PyTree) -> int:
    """Count the number of trainable parameters in a model.

    Under the hood this just counts the number of inexact
    JAX/NumPy array elements in the pytree that are not
    wrapped by [`klax.NonTrainable`][].

    Warning:
        If you use `jax.lax.stop_gradient` or any other method
        besides [`klax.NonTrainable`][] to make arrays not receive
        gradient updates, then this function will overestimate
        the number of trainable parameters!
        Consider using [`klax.NonTrainable`][] or counting the trainable
        parameters manually.

    Args:
        model: Arbitrary pytree.

    Returns:
        Integer count of inexact inexact JAX/NumPy array elements.

    """
    return sum(
        jax.tree.flatten(
            jax.tree.map(
                lambda x: x.size
                if eqx.is_inexact_array(x) and not isinstance(x, NonTrainable)
                else 0,
                model,
                is_leaf=lambda x: isinstance(x, NonTrainable),
            )
        )[0]
    )
