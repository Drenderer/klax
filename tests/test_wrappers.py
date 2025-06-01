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

from math import prod

import equinox as eqx
from klax import (
    apply,
    contains_constraints,
    contains_unwrappables,
    NonNegative,
    NonTrainable,
    non_trainable,
    Parameterize,
    Symmetric,
    SkewSymmetric,
    unwrap,
    finalize,
)
import jax
import jax.random as jr
import jax.numpy as jnp
import pytest


def test_nested_unwrap():
    param = Parameterize(
        jnp.square,
        Parameterize(jnp.square, Parameterize(jnp.square, 2)),
    )
    assert unwrap(param) == jnp.square(jnp.square(jnp.square(2)))


def test_parameterize():
    diag = Parameterize(jnp.diag, jnp.ones(3))
    assert jnp.allclose(jnp.eye(3), unwrap(diag))


def test_non_trainable(getarraywrap):
    # Array model
    model = non_trainable((jnp.ones(3), 1))

    def loss(model):
        model = unwrap(model)
        return model[0].sum()

    grad = eqx.filter_grad(loss)(model)[0].tree
    assert grad.shape == (3,)
    assert jnp.all(grad == 0.0)

    # Constraint model
    model = non_trainable((getarraywrap(jnp.ones(3)), 1))
    grad = eqx.filter_grad(loss)(model)[0].tree.parameter
    assert grad.shape == (3,)
    assert jnp.all(grad == 0.0)


def test_non_negative(getkey):
    # Negative array input
    parameter = -jr.uniform(getkey(), (10,))
    non_neg = NonNegative(parameter)
    assert jnp.all(finalize(non_neg) == 0)
    assert jnp.all(apply(non_neg).parameter == 0)

    # Positive array input
    parameter = jr.uniform(getkey(), (10,))
    non_neg = NonNegative(parameter)
    assert jnp.all(finalize(non_neg) == parameter)
    assert jnp.all(apply(non_neg).parameter == parameter)


def test_symmetric(getkey):
    # Constraint
    parameter = jr.normal(getkey(), (3, 10, 3, 3))
    symmetric = Symmetric(parameter)
    _symmetric = unwrap(symmetric)
    assert _symmetric.shape == parameter.shape
    assert jnp.array_equal(
        _symmetric, jnp.transpose(_symmetric, axes=(0, 1, 3, 2))
    )


def test_skewsymmetric(getkey):
    # Constraint
    parameter = jr.normal(getkey(), (3, 10, 3, 3))
    symmetric = SkewSymmetric(parameter)
    _symmetric = unwrap(symmetric)
    assert _symmetric.shape == parameter.shape
    assert jnp.array_equal(
        _symmetric, -jnp.transpose(_symmetric, axes=(0, 1, 3, 2))
    )


test_cases = {
    "NonTrainable": lambda key: NonTrainable(jr.normal(key, 10)),
    "Parameterize-exp": lambda key: Parameterize(jnp.exp, jr.normal(key, 10)),
    "NonNegative": lambda key: NonNegative(jr.normal(key, 10)),
}


@pytest.mark.parametrize("shape", [(), (2,), (5, 2, 4)])
@pytest.mark.parametrize(
    "wrapper_fn", test_cases.values(), ids=test_cases.keys()
)
def test_vectorization_invariance(wrapper_fn, shape):
    keys = jr.split(jr.key(0), prod(shape))
    wrapper = wrapper_fn(keys[0])  # Standard init

    # Multiple vmap init - should have same result in zero-th index
    vmap_wrapper_fn = wrapper_fn
    for _ in shape:
        vmap_wrapper_fn = eqx.filter_vmap(vmap_wrapper_fn)

    vmap_wrapper = vmap_wrapper_fn(keys.reshape(shape))

    unwrapped = unwrap(wrapper)
    unwrapped_vmap = unwrap(vmap_wrapper)
    unwrapped_vmap_zero = jax.tree.map(
        lambda leaf: leaf[*([0] * len(shape)), ...],
        unwrapped_vmap,
    )
    assert eqx.tree_equal(unwrapped, unwrapped_vmap_zero, atol=1e-7)


def test_contains():
    # Contains nothing
    tree = (1, "abc", jnp.ones(3))
    assert not contains_unwrappables(tree)
    assert not contains_constraints(tree)

    # Contains Unwrappable
    tree = (1, "abc", Parameterize(jnp.square, jnp.ones(3)))
    assert contains_unwrappables(tree)
    assert not contains_constraints(tree)

    # Contains Constraint
    tree = (1, "abc", NonNegative(jnp.ones(3)))
    assert contains_unwrappables(tree)
    assert contains_constraints(tree)
