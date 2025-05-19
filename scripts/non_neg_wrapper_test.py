"""
This  script shows that the old NonNegative wrapper does not work as expected
when the optimizer drives the wrapped parameter into the negative region. Then
the gradients for the parameter are stopped by ReLU and the parameter will never
recieve and updates again.
The new NonNegative wrapper uses softplus to ensure non-negativity, eliviating
this issue.
"""

# %% Imports
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from matplotlib import pyplot as plt

from klax import fit, ParameterWrapper, NonNegative, Positive, unwrap, HistoryCallback


# %% Define old NonNegative Wrapper
class OldNonNegative(ParameterWrapper):
    """Applies a non-negative constraint.

    Args:
        parameter: The parameter that is to be made non-negative. It can either
            be a `jax.Array` or a `ParameterWrapper`.
    """

    parameter: Array

    def __init__(self, parameter: Array | ParameterWrapper):
        # Ensure that the parameter fulfills the constraint initially
        self.parameter = self._non_neg(unwrap(parameter))

    def _non_neg(self, x: Array) -> Array:
        return jnp.maximum(x, 0)

    def unwrap(self) -> Array:
        return self._non_neg(self.parameter)


class ParameterHistory(HistoryCallback):
    parameters: list
    weights: list

    def __init__(self, log_every=100, verbose=False):
        super().__init__(log_every, verbose)
        self.parameters = []
        self.weights = []

    def __call__(self, cbargs):
        super().__call__(cbargs)
        if cbargs.step % self.log_every == 0:
            self.parameters.append(cbargs.model.weight.parameter)
            self.weights.append(unwrap(cbargs.model).weight)


# %% Define a simple model
class SimpleModel(eqx.Module):
    weight: jax.Array

    def __init__(self, wrapper: type[ParameterWrapper]):
        self.weight = wrapper(jnp.array(1.0))

    def __call__(self, x):
        return self.weight * x


# %% Generate data which requires the parameter to be negative
key = jr.key(0)

x = jnp.linspace(-1, 1, 100)
y1 = jax.vmap(lambda x: -2 * x)(x)
y2 = jax.vmap(lambda x: 2 * x)(x)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for wrapper in [OldNonNegative, NonNegative, Positive]:
    model = SimpleModel(wrapper)

    print(
        f"{wrapper.__name__}: Initial weights\nunwrapped weight: {unwrap(model).weight}, parameter: {model.weight.parameter}"
    )

    model, hist = fit(model, (x, y1), steps=5000, history=ParameterHistory(), key=key)
    model, hist = fit(model, (x, y2), steps=5000, history=hist, key=key)

    print(
        f"{wrapper.__name__}: Final weights\nunwrapped weight: {unwrap(model).weight}, parameter: {model.weight.parameter}"
    )

    axes[0].plot(hist.steps, hist.parameters, label=wrapper.__name__)
    axes[1].plot(hist.steps, hist.weights, label=wrapper.__name__)

# Plot the parameters over the steps
for ax in axes:
    ax.axhline(0.0, xmin=0.0, ls="--", c="grey")
    ax.legend()

axes[0].set(title="Parameter Hisotry", xlabel="steps", ylabel="weight.parameter")
axes[1].set(title="Weight Hisotry", xlabel="steps", ylabel="weight")

plt.tight_layout()
plt.show()
