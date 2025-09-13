from abc import abstractmethod
from typing import Protocol

import equinox as eqx
import jax
import optax
from jaxtyping import PyTree

from ._losses import ValueAndGradFn, ValueFn


class Updater(Protocol):
    @abstractmethod
    def __call__(
        self, model: PyTree, batch: PyTree, opt_state: PyTree
    ) -> tuple[PyTree, PyTree]:
        pass


class UpdaterFactory(Protocol):
    @abstractmethod
    def __call__(
        self,
        opt_update: optax.TransformUpdateFn | optax.TransformUpdateExtraArgsFn,
        value_fn: ValueFn,
        value_and_grad_fn: ValueAndGradFn,
    ) -> Updater:
        pass


def optax_transform_update_fn_updater(
    opt_update: optax.TransformUpdateFn,
    value_fn: ValueFn,
    value_and_grad_fn: ValueAndGradFn,
) -> Updater:
    def wrapper(model, batch, opt_state):
        _, grad = value_and_grad_fn(model, batch)
        updates, opt_state = opt_update(
            grad,
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    return wrapper


def optax_transform_update_fn_extra_args_updater(
    opt_update: optax.TransformUpdateExtraArgsFn,
    value_fn: ValueFn,
    value_and_grad_fn: ValueAndGradFn,
) -> Updater:
    def wrapper(model, batch, opt_state):
        value, grad = value_and_grad_fn(model, batch)
        updates, opt_state = opt_update(
            grad,
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
            value=value,
            grad=grad,
            value_fn=jax.tree_util.Partial(value_fn, model=model, batch=batch),
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    return wrapper
