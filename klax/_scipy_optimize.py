import itertools
from collections.abc import Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree, PyTreeDef
from scipy.optimize import OptimizeResult, minimize

from ._callbacks import Callback
from ._losses import Loss
from ._wrappers import Constraint, NonNegative, NonTrainable


def _is_static_leaf(element: Any) -> bool:
    """Return true if `element` should be considered as leaf for finding static parts of a model."""
    return isinstance(element, NonTrainable)


def _is_trainable(element: Any) -> bool:
    """Return `True` if `element` should be treated as trainable."""
    if _is_static_leaf(element):
        return False
    return eqx.is_inexact_array(element)


def _is_constraint(element: PyTree) -> bool:
    """Return True if the `element` is a klax.Constraint."""
    return isinstance(element, Constraint)


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


class ScipyModelAdapter[T: PyTree]:
    """Adapter to translate an equinox model into the format of scipy minimize.

    The adapter provides functionality to convert an equinox model into a single
    vector of trainable parameters and back.
    Additionally, the bounds (allowed range) for each element in the vector are
    computed and stored by the adapter. This is then used to inform scipy of
    potential (non-negativity) constraints.
    """

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
            model, _is_trainable, is_leaf=_is_static_leaf
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
            model, _is_trainable, is_leaf=_is_static_leaf
        )
        leafs, _ = jax.tree.flatten(params)
        return jnp.concat(jax.tree.map(lambda x: x.flatten(), leafs))

    def unflatten(self, x: Array) -> T:
        """Unflatten a flat array of parameters into the original model.

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


class ScipyTrainingState:
    """`TrainingState` mock-up class compatible with `scipy_fit`."""

    _model: PyTree
    opt_state: None
    _adapter: ScipyModelAdapter
    _run_state: PyTree
    _x: np.ndarray

    def __init__(
        self, adapter: ScipyModelAdapter, run_state: PyTree, x: np.ndarray
    ) -> None:
        self._adapter = adapter
        self._run_state = run_state
        self._x = x
        self.opt_state = None

    @property
    def model(self) -> PyTree:
        return self._adapter.unflatten(self._x)

    @property
    def run_state(self) -> PyTree:
        return self._run_state

    def update(self, xk: np.ndarray):
        self._x = xk


class ScipyTrainingContext:
    """`TrainingContext` mock-up class compatible with `scipy_fit`."""

    state: ScipyTrainingState
    optimizer: str
    loss: Loss
    step: int
    steps: int

    def __init__(
        self,
        adapter: ScipyModelAdapter,
        run_state: PyTree,
        optimizer: str,
        max_steps: int,
        loss: Loss,
        x: np.ndarray,
    ):
        self.state = ScipyTrainingState(adapter, run_state, x)
        self.optimizer = optimizer
        self.loss = loss
        self.step = 0
        self.steps = max_steps

    @property
    def batch_generator(self):
        raise ValueError("There exists no `batch_generator` for `scipy_fit`.")

    def update(self, xk: np.ndarray):
        self.state.update(xk)
        self.step += 1


class ScipyCallbackAdapter:
    """Combine multiple callbacks into one, scipy compatible callback."""

    callbacks: Sequence[Callback]
    context: ScipyTrainingContext

    def __init__(
        self,
        callbacks: Sequence[Callback],
        adapter: ScipyModelAdapter,
        run_state: PyTree,
        optimizer: str,
        max_steps: int,
        loss: Loss,
        x: np.ndarray,
    ):
        self.callbacks = callbacks
        self.context = ScipyTrainingContext(
            adapter, run_state, optimizer, max_steps, loss, x
        )

    def on_training_start(self):
        for callback in self.callbacks:
            callback.on_training_start(self.context)

    def on_training_step(self, xk: np.ndarray):
        self.context.update(xk)
        stop = False
        for callback in self.callbacks:
            stop |= bool(callback.on_training_step(self.context))
        if stop:
            raise StopIteration

    def on_training_end(self):
        for callback in self.callbacks:
            callback.on_training_end(self.context)


def scipy_loss_wrapper(loss: Loss, adapter: ScipyModelAdapter, data: PyTree):
    """Transform a [`Loss`][klax.Loss] into a scipy minimize objective function.

    Args:
        loss: [`Loss`][klax.Loss] to wrap.
        adapter: ScipyModelAdapter for the model.
        data: Training data.

    Returns:
        Objective function compatible with `scipy.minimize`.

    """

    @jax.jit
    def _jitted_wrapped_loss(x, run_state):
        model = adapter.unflatten(x)
        value, grad = loss.value_and_grad(model, data, run_state)
        grad = adapter.flatten(grad)
        return value, grad

    def wrapped_loss(x, run_state):
        value, grad = _jitted_wrapped_loss(x, run_state)
        value = np.array(value, dtype=np.float64)
        grad = np.array(grad, dtype=np.float64)
        return value, grad

    return wrapped_loss


def scipy_fit[T: PyTree](
    model: T,
    data: PyTree[Any],
    *,
    loss: Loss,
    optimizer: Literal["L-BFGS-B", "SLSQP"] = "SLSQP",
    options: dict | None = None,
    run_state: PyTree[Any] = None,
    callbacks: Sequence[Callback] | None = None,
) -> tuple[T, OptimizeResult]:
    """Fit a model using scipy's [minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#rdd2e1855725e-12).

    In contrast to [`fit`][klax.fit] this enables the use of *second
    order optimizers*, which can be very beneficial for smaller
    models.

    Internally, this function translates from the equinox model formulation to a
    flat vector of trainable variables with assigned min/max bounds. The loss
    function is then wrapped to accept this vector of design parameters and
    evaluate the model on the entire dataset. The wrapped loss and bounds are
    passed to `scipy.optimize.minimize`.
    Currently, only box constraints in the form of the bounds are supported;
    (in-)equality constraints are *not* supported.

    !!! Note
        This method optimizes the loss function evaluated on the entire dataset.
        For very large datasets, this may become inefficient.

    !!! Warning
        **Compatibility**: Many klax functionalities are not, or only partially
        compatible with `scipy_fit`.

    !!! Warning
        **Frozen parameters**: To ensure that a parameter is not updated by
        `scipy_fit` wrap it with [`NonTrainable`][klax.NonTrainable]
        (see also [`non_trainable`][klax.non_trainable]).
        Just blocking gradients with `jax.lax.stop_gradient` is potentially not
        sufficient.

    Args:
        model: The model instance, which should be trained. It must be a
            subclass of `equinox.Module`. The model may contain
            [`klax.Unwrappable`][] wrappers.
        data: The training data can be any `PyTree` with at least some
            `ArrayLike` leaves. Most likely you'll want `data` to be a
            tuple `(x, y)` with model inputs `x` and model outputs `y`.
        loss: The [loss][klax.Loss] function.
            Defaults to `mse`.
        optimizer: Type of solver. Available options: `"L-BFGS-B"` and
            `"SLSQP"`.
            Defaults to `"SLSQP"`.
        options: Dict of solver specific options passed to
            `scipy.optimize.minimize`.
            All solvers accept:

            - `"maxiter"` (int): Maximum number of iterations to perform.
                Depending on the method each iteration may use several function
                evaluations.
                If `options` does not contain `"maxiter"` or `options=None` then
                klax uses the default `maxiter=1000`.
            - `"disp"` (bool): Set to True to print convergence messages.

            For the solver-specific options see [SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb)
            and [SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp).
        run_state: Auxiliary runtime state, that is passed to the loss function.
            Can be updated via callbacks.
            Defaults to `None`.
        callbacks: List of [Callbacks][klax.Callback]. They can be used to
            implement early stopping, custom logging and more.
            !!! Warning
                Not all functionality of [Callbacks][klax.Callback] is available
                for `scipy_fit`.
            Defaults to `None`.

    Returns:
        Fitted model and scipy's [`OptimizeResult`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult).

    """
    options = dict() if options is None else options
    max_steps = options.setdefault("maxiter", 1000)

    adapter = ScipyModelAdapter(model)
    scipy_loss_and_grad = scipy_loss_wrapper(loss, adapter, data)
    x0 = adapter.flatten(model)

    callbacks = [] if callbacks is None else list(callbacks)
    callback = ScipyCallbackAdapter(
        callbacks, adapter, run_state, optimizer, max_steps, loss, x0
    )

    callback.on_training_start()

    optimize_result = minimize(
        fun=scipy_loss_and_grad,
        x0=x0,
        args=run_state,
        jac=True,
        method=optimizer,
        options=options,
        bounds=adapter.bounds,
        callback=callback.on_training_step,
    )

    callback.on_training_end()

    model = adapter.unflatten(optimize_result.x)
    return model, optimize_result
