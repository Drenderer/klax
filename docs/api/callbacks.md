---
title: Callbacks
---

Callbacks allow users to inject custom behavior into the
training loop *after* each parameter update and *outside* of jit. This can be used for logging, early stopping, modifying the model in a jax-incompatible way and much more!

::: klax.Callback
    options:
        members:
            - on_training_start
            - on_training_step
            - on_training_end

## Example

```python
import equinox as eqx
import jax
import jax.numpy as jnp

import klax


class PrintWeightsEvery(klax.Callback):
    """Simple callback that prints the values of the weight matrix every x steps during training."""

    def __init__(self, every: int = 100):
        self.every = every

    def on_training_start(self, context):
        print(
            f"Initially, the  weight matrix is: \n{context.state.model.weight}."
        )

    def on_training_step(self, context: klax.TrainingContext) -> bool | None:
        step = context.step
        if step % self.every == 0:
            model = context.state.model
            print(f"At step {context.step} the weight is: \n{model.weight}.")
        return None

    def on_training_end(self, context):
        print(
            f"Training has finished! This is the final weight matrix: \n{context.state.model.weight}."
        )

# Initialize the callback
my_callback = PrintWeightsEvery(100)

# Create a simple linear layer and some dummy data
key = jax.random.PRNGKey(0)
model = eqx.nn.Linear(in_features=4, out_features=1, key=key)

x = jax.random.normal(key, (256, 4))
y = jnp.sum(x, axis=1, keepdims=True)

# Run the training loop with the callback
trained_model, history = klax.fit(
    model,
    data=(x, y),
    steps=500,
    callbacks=[my_callback], # <- Tell the training loop about the callback.
    verbose=0, # We don't use the default logging here, since this interferes with printing the weights.
    key=key,
)
```

Output:
```python
>>> Initially, the  weight matrix is: 
>>> [[ 0.34231412 -0.31762135 -0.2728219  -0.37927437]].
>>> At step 100 the weight is: 
>>> [[ 0.42761046 -0.22412491 -0.17941569 -0.2861022 ]].
>>> At step 200 the weight is: 
>>> [[ 0.5072705  -0.13292825 -0.08987902 -0.19596328]].
>>> At step 300 the weight is: 
>>> [[ 0.5813759  -0.04522126 -0.00425893 -0.10924208]].
>>> At step 400 the weight is: 
>>> [[ 0.6492207   0.0390796   0.07759552 -0.02558598]].
>>> At step 500 the weight is: 
>>> [[0.71085125 0.1197266  0.15574388 0.05466396]].
>>> Training has finished! This is the final weight matrix: 
>>> [[0.71085125 0.1197266  0.15574388 0.05466396]].
```