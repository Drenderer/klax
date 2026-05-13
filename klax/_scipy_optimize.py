import itertools
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree, PyTreeDef
from scipy.optimize import OptimizeResult, minimize

from ._losses import Loss, mse
from ._wrappers import Constraint, NonNegative, NonTrainable


def _is_static_leaf(element: Any) -> bool:
    """Return true if the `element` should be treated as static."""
    return isinstance(element, NonTrainable)


def _is_static(element: Any) -> bool:
    """Return `True` if `element` should be treated as static."""
    if _is_static_leaf(element):
        return True
    else:
        return eqx.is_inexact_array(element)


def _get_bounds(element: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of bounds for each `element`."""
    if isinstance(element, NonNegative):
        size = element.unwrap().size
        return size * [(0, np.inf)]
    if isinstance(element, Array):
        size = element.size
        return size * [(-np.inf, np.inf)]
    raise ValueError(
        f"I don't know how to compute bounds for type {type(element)}"
    )


def _is_constraint(element: PyTree) -> bool:
    """Return True if the `element` is a klax.Constraint."""
    return isinstance(element, Constraint)


class ScipyModelAdapter[T: PyTree]:
    """Adapter to translate an equinox model into the scipy minimize formulation."""

    static: PyTree
    tree_def: PyTreeDef  # pyright: ignore[reportInvalidTypeForm]
    shapes: list[tuple[int]]
    split_indices: tuple[int]
    bounds: list[tuple[np.ndarray, np.ndarray]]

    def __init__(self, model: T) -> None:
        """Initialize the adapter.

        Args:
            model: Model to adapt for scipy.

        """
        params, static = eqx.partition(
            model, _is_static, is_leaf=_is_static_leaf
        )
        leafs, tree_def = jax.tree.flatten(params)

        shapes = jax.tree.map(lambda x: x.shape, leafs)
        sizes = jax.tree.map(lambda x: x.size, leafs)
        split_indices = tuple(jnp.cumsum(jnp.stack(sizes)[:-1]))

        bounds, _ = jax.tree.flatten(params, is_leaf=_is_constraint)
        bounds = jax.tree.map(_get_bounds, bounds, is_leaf=_is_constraint)
        bounds = list(itertools.chain.from_iterable(bounds))

        self.static = static
        self.tree_def = tree_def
        self.shapes = shapes
        self.split_indices = split_indices
        self.bounds = bounds

    def flatten(self, model: T) -> Array:
        """Return a flat array of trainable parameters for the model.

        Args:
            model: Model to flatten.

        Returns:
            Flat array of model parameters.

        """
        params, _ = eqx.partition(
            model, eqx.is_inexact_array, is_leaf=_is_static
        )
        leafs, _ = jax.tree.flatten(params)
        return jnp.concat(jax.tree.map(lambda x: x.flatten(), leafs))

    def unflatten(self, x: Array) -> T:
        """Unflatten a flat array of paramters into the original model.

        Args:
            x: Flat array of model parameters.

        Returns:
            Unflattened model.

        """
        flat_params = jnp.split(x, self.split_indices)
        leafs = jax.tree.map(
            lambda x, s: x.reshape(s), flat_params, self.shapes
        )
        params = jax.tree.unflatten(self.tree_def, leafs)
        return eqx.combine(params, self.static)


def scipy_loss_wrapper(loss: Loss, converter: ScipyModelAdapter, data: PyTree):
    """Transform a [`Loss`][klax.Loss] into a scipy minimize objective function.

    Args:
        loss: [`Loss`][klax.Loss] to wrap.
        converter: ScipyModelAdapter for the model.
        data: Training data.

    Returns:
        Objective function compatible with `scipy.minimize`.

    """

    @jax.jit
    def _jitted_wrapped_loss(x, run_state):
        model = converter.unflatten(x)
        value, grad = loss.value_and_grad(model, data, run_state)
        grad = converter.flatten(grad)
        return value, grad

    def wrapped_loss(x, run_state):
        value, grad = _jitted_wrapped_loss(x, run_state)
        return np.array(value, dtype=np.float64), np.array(grad, dtype=np.float64)

    return wrapped_loss


def scipy_fit[T: PyTree](
    model: T,
    data: PyTree[Any],
    *,
    run_state: PyTree[Any] = None,
    max_steps: int = 1000,
    loss: Loss = mse,
    optimizer: Literal[
        "Nelder-Mead",
        "L-BFGS-B",
        "SLSQP",
        "Powell",
        "trust-constr",
        "COBYLA",
        "COBYQA",
    ] = "SLSQP",
    tol: float = 1e-12,
    verbose: bool = False,
) -> tuple[T, OptimizeResult]:
    """Fit a model using scipy's minimize.

    In contrast to the default fit this enables the use of second
    order optimizers, which can be very beneficial for smaller
    models.

    !!! Warning
    Most klax functionalities are not compatible with `scipy_fit`.

    Args:
        model: The model instance, which should be trained. It must be a
            subclass of `equinox.Module`. The model may contain
            [`klax.Unwrappable`][] wrappers.
        data: The training data can be any `PyTree` with at least some
            `ArrayLike` leaves. Most likely you'll want `data` to be a
            tuple `(x, y)` with model inputs `x` and model outputs `y`.
        run_state: Auxiliary runtime state, that is passed to the loss function.
            Can be updated via callbacks.
            Defaults to `None`.
        max_steps: Maximum number of iterations to perform. Depending on the
            method each iteration may use several function evaluations.
        loss: The [loss][klax.Loss] function.
            Defaults to `mse`.
        optimizer: Type of solver. Available options:
            - `"Nelder-Mead"`
            - `"L-BFGS-B"`
            - `"SLSQP"`
            - `"Powell"`
            - `"trust-constr"`
            - `"COBYLA"`
            - `"COBYQA"`
            Defaults to `"SLSQP"`.
        tol: Tolerance for termination.
        verbose: Set to True to print convergence messages.


    Returns:
        Fitted model and `scipy.OptimizeResult`.

    """
    adapter = ScipyModelAdapter(model)
    scipy_loss_and_grad = scipy_loss_wrapper(loss, adapter, data)
    x0 = adapter.flatten(model)

    optimize_result = minimize(
        fun=scipy_loss_and_grad,
        x0=x0,
        args=run_state,
        jac=True,
        tol=tol,
        method=optimizer,
        options={"maxiter": max_steps, "ftol": tol, "disp": verbose},
        bounds=adapter.bounds,
        constraints=(),
    )
    model = adapter.unflatten(optimize_result.x)

    return model, optimize_result
