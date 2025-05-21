# Copyright 2025 The Klax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import equinox as eqx
import klax
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array
import optax
import pytest
from typing import Self

from klax import AbstractUnwrappable, ArrayWrapper, unwrap


def test_training(getkey):
    # Fitting a linear function
    x = jnp.linspace(0.0, 1.0, 2).reshape(-1, 1)
    y = 2.0 * x + 1.0
    model = eqx.nn.Linear(1, 1, key=getkey())
    model, _ = klax.fit(model, (x, y), optimizer=optax.adam(1.0), key=getkey())
    y_pred = jax.vmap(model)(x)
    assert jnp.allclose(y_pred, y)

    # Multiple inputs
    class Model(eqx.Module):
        weight: Array

        def __call__(self, x):
            b, x = x
            return b + self.weight * x

    x = jrandom.uniform(key=getkey(), shape=(10,))
    b = 2.0
    y = b + 2 * x
    model = Model(weight=jnp.array(1.0))
    model, _ = klax.fit(
        model,
        ((b, x), y),
        batch_axis=0,  # Test automatic batch axis braodcasting to data
        optimizer=optax.adam(1.0),
        key=getkey(),
    )
    y_pred = jax.vmap(model, in_axes=((None, 0),))((b, x))
    assert jnp.allclose(y_pred, y)

    # Continued training with history and solver state
    x = jrandom.uniform(getkey(), (2, 1))
    model = eqx.nn.Linear(1, 1, key=getkey())
    history = klax.HistoryCallback(log_every=100)
    model, history = klax.fit(model, (x, x), steps=1000, history=history, key=getkey())
    assert len(history.steps) == 11
    assert len(history.loss) == 11
    time_1 = history.training_time
    model, history = klax.fit(
        model,
        (x, x),
        steps=500,
        history=history,
        init_opt_state=history.last_opt_state,
        key=getkey(),
    )
    assert len(history.steps) == 16
    assert len(history.loss) == 16
    assert history.steps[-1] == 1500
    time_2 = history.training_time
    assert time_1 < time_2

    # Validation data
    x = jrandom.uniform(getkey(), (2, 1))
    model = eqx.nn.Linear(1, 1, key=getkey())
    _, history = klax.fit(model, (x, x), validation_data=(x, x), key=getkey())
    assert len(history.val_loss) == 11

    # Callbacks
    x = jrandom.uniform(getkey(), (2, 1))
    model = eqx.nn.Linear(1, 1, key=getkey())

    class MyCallback(klax.Callback):
        def __call__(self, cbargs: klax.CallbackArgs):
            if cbargs.step == 123:
                return True

    _, history = klax.fit(
        model,
        (x, x),
        history=klax.HistoryCallback(1),
        callbacks=(MyCallback(),),
        key=getkey(),
    )
    print(history.log_every)
    assert history.steps[-1] == 123


@pytest.mark.parametrize("optimizer", [
        optax.adabelief(1.0),
        optax.adadelta(1.0),
        optax.adan(1.0),
        optax.adafactor(1.0),
        optax.adagrad(1.0),
        optax.adam(1.0),
        optax.adamw(1.0),
        optax.adamax(1.0),
        optax.adamaxw(1.0),
        optax.amsgrad(1.0),
        optax.fromage(1.0),
        optax.lamb(1.0),
        optax.lars(1.0),
        optax.lbfgs(1.0),
        optax.lion(1.0),
        optax.nadam(1.0),
        optax.nadamw(1.0),
        optax.noisy_sgd(1.0),
        optax.novograd(1.0),
        optax.optimistic_gradient_descent(1.0),
        optax.optimistic_adam(1.0),
        optax.polyak_sgd(1.0),
        optax.radam(1.0),
        optax.rmsprop(1.0),
        optax.sgd(1.0),
        optax.sign_sgd(1.0),
        optax.sm3(1.0),
        optax.yogi(1.0),
    ]
)
def test_training_optax_optimizers(getkey, optimizer):
    # Test all optex optimizers
    x = jrandom.uniform(getkey(), (2, 1))
    model = eqx.nn.Linear(1, 1, key=getkey())
    klax.fit(model, (x, x), optimizer=optimizer, key=getkey())


def test_apply_in_training(getkey):

    # Create dummy data
    x = jnp.linspace(0.0, 1.0, 20)
    y = -2 * x - 1

    # Create dummy ArrayWrapper
    class AtLeast(ArrayWrapper):
        array: Array
        minval: Array        

        def unwrap(self) -> Array:
            return self.array
        
        def apply(self) -> Self:
            return eqx.tree_at(
                lambda x: x.array,
                self,
                replace=jnp.maximum(self.array, self.minval),
            )

    # Create dummy model
    class Model(eqx.Module):
        weight: AbstractUnwrappable[Array]
        bias: AbstractUnwrappable[Array]

        def __init__(self):
            self.weight = AtLeast(jnp.array(0.), jnp.array(-1))
            self.bias = AtLeast(jnp.array(-1.), jnp.array(0))

        def __call__(self, x):
            return self.weight * x + self.bias
        
    # Create and train model
    model = Model()
    model, _ = klax.fit(model, (x, y), key=getkey())

    model_ = unwrap(model)
    assert model_.weight >= -1
    assert model_.bias >= 0



