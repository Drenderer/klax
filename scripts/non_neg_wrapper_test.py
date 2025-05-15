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

from klax import fit, ParameterWrapper, NonNegative, unwrap, HistoryCallback
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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

    def __init__(self, log_every=100, verbose=False):
        super().__init__(log_every, verbose)
        self.parameters = []

    def __call__(self, cbargs):
        super().__call__(cbargs)
        if cbargs.step % self.log_every == 0:
            self.parameters.append(cbargs.model.weight.parameter)


# %% Define a simple model
class SimpleModel(eqx.Module):
    weight: jax.Array

    def __init__(self, wrapper: type[ParameterWrapper]):
        self.weight = wrapper(jnp.array(-1.0))

    def __call__(self, x):
        return self.weight * x


# %% Generate data which requires the parameter to be negative
x = jnp.linspace(-1, 1, 100)
y = jax.vmap(lambda x: -2 * x)(x)

# Define the models
model_old = SimpleModel(wrapper=OldNonNegative)
model = SimpleModel(wrapper=NonNegative)

key = jr.key(0)

# Let the optimizer run the parameter into the negatives
print(
    f"Initial weights: \nOld wrapper: unwrapped weight: {unwrap(model_old).weight}, parameter: {model_old.weight.parameter}\nNew wrapper: weight: {unwrap(model).weight}, parameter: {model.weight.parameter}"
)
model_old, hist_old = fit(
    model_old, (x, y), steps=500, history=ParameterHistory(), key=key
)
model, hist = fit(model, (x, y), steps=500, history=ParameterHistory(), key=key)

# Redefine the data so the parameter should be positive
y = jax.vmap(lambda x: 2 * x)(x)

# Train the models to test recovery
model_old, hist_old = fit(model_old, (x, y), steps=5000, history=hist_old, key=key)
model, hist = fit(model, (x, y), steps=5000, history=hist, key=key)
print(
    f"Final weights: \nOld wrapper: unwrapped weight: {unwrap(model_old).weight}, parameter: {model_old.weight.parameter}\nNew wrapper: weight: {unwrap(model).weight}, parameter: {model.weight.parameter}"
)


# %% Plot the parameters over the steps
fig, ax = plt.subplots()
ax.axhline(0.0, xmin=0.0, ls="--", c="grey")
ax.plot(hist.steps, hist.parameters, label="New")
ax.plot(hist_old.steps, hist_old.parameters, label="Old")
ax.set(title="Parameter Hisotry", xlabel="steps", ylabel="weight.parameter")

# Add a zoom-in lens to focus on the origin

# Create an inset axis
ax_inset = inset_axes(ax, width="30%", height="30%", loc="upper left")
ax_inset.plot(hist.steps, hist.parameters, label="New")
ax_inset.plot(hist_old.steps, hist_old.parameters, label="Old")
ax_inset.axhline(0.0, xmin=0.0, ls="--", c="grey")
ax_inset.set_xlim(-100, 300)  # Adjust limits to zoom into the origin
ax_inset.set_ylim(-0.1, 0.1)  # Adjust limits to zoom into the origin
ax_inset.set_xticks([])
ax_inset.set_yticks([])

# Mark the inset on the main plot
mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

ax.legend(loc="lower right")
plt.show()
