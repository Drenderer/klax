
# %% Imports
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree
import equinox as eqx
from matplotlib import pyplot as plt

from klax import ParameterWrapper, fit, Callback, TrainingState, HistoryCallback


# %% Define a simple model
class SimpleModel(eqx.Module):
    weight: Array

    def __init__(self, wrapper: type[ParameterWrapper]):
        self.weight = wrapper(jnp.array(-1.))

    def __call__(self, x):
        return self.weight * x
    
class NonNegativeConstraint(ParameterWrapper):
    parameter: Array
    def unwrap(self):
        return self.parameter
    
    def apply(self):
        return NonNegativeConstraint(jnp.maximum(self.parameter, 0))

def apply(tree: PyTree):
    def _unwrap(tree, *, include_self: bool):
        def _map_fn(leaf):
            if isinstance(leaf, NonNegativeConstraint):
                # Apply subnodes, then itself
                return _unwrap(leaf, include_self=False).apply()
            return leaf

        def is_leaf(x):
            is_unwrappable = isinstance(x, NonNegativeConstraint)
            included = include_self or x is not tree
            return is_unwrappable and included

        return jax.tree_util.tree_map(f=_map_fn, tree=tree, is_leaf=is_leaf)

    return _unwrap(tree, include_self=True)

class ApplyCallback(Callback):
    def __call__(self, state: TrainingState):
        state.model = apply(state.model)


class ParameterHistory(HistoryCallback):
    parameters: list

    def __init__(self, log_every=100, verbose=False):
        super().__init__(log_every, verbose)
        self.parameters = []

    def __call__(self, state):
        super().__call__(state)
        if state.step % self.log_every == 0:
            self.parameters.append(state.model.weight.parameter)

# %% Test

model = SimpleModel(NonNegativeConstraint)

key = jr.key(0)

x = jnp.linspace(-1, 1, 100)
y_1 = jax.vmap(lambda x: -2 * x)(x)
y_2 = jax.vmap(lambda x: 2 * x)(x)

model, hist = fit(model, (x, y_1), callbacks=[ApplyCallback()], history=ParameterHistory(), key=key)
model, hist = fit(model, (x, y_2), callbacks=[ApplyCallback()], history=hist, key=key)

plt.plot(hist.steps, hist.parameters)
plt.show()