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

import typing
from abc import abstractmethod
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar


@typing.runtime_checkable
class ValueFn(Protocol):
    @abstractmethod
    def __call__(self, model: PyTree, data: PyTree) -> Scalar:
        pass


@typing.runtime_checkable
class ValueAndGradFn(Protocol):
    @abstractmethod
    def __call__(self, model: PyTree, data: PyTree) -> tuple[Scalar, PyTree]:
        pass


class LossFactory(Protocol):
    @abstractmethod
    def __call__(self, batch_axis) -> tuple[ValueFn, ValueAndGradFn]:
        pass


def mse(batch_axis) -> tuple[ValueFn, ValueAndGradFn]:
    def value_fn(model: PyTree, data: PyTree) -> Scalar:
        x, y = data
        if isinstance(batch_axis, tuple):
            in_axes = batch_axis[0]
        else:
            in_axes = batch_axis
        y_pred = eqx.filter_vmap(model, in_axes=(in_axes,))(x)
        return jnp.mean(jnp.square(y_pred - y))

    def value_and_grad_fn(
        model: PyTree, data: PyTree
    ) -> tuple[Scalar, PyTree]:
        return eqx.filter_value_and_grad(value_fn)(model, data)

    return value_fn, value_and_grad_fn


def mae(batch_axis) -> tuple[ValueFn, ValueAndGradFn]:
    def value_fn(model: PyTree, data: PyTree) -> Scalar:
        x, y = data
        if isinstance(batch_axis, tuple):
            in_axes = batch_axis[0]
        else:
            in_axes = batch_axis
        y_pred = eqx.filter_vmap(model, in_axes=(in_axes,))(x)
        return jnp.mean(jnp.abs(y_pred - y))

    def value_and_grad_fn(
        model: PyTree, data: PyTree
    ) -> tuple[Scalar, PyTree]:
        return eqx.filter_value_and_grad(value_fn)(model, data)

    return value_fn, value_and_grad_fn
