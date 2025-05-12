from __future__ import annotations
from typing import Optional, Callable

import jax
from jaxtyping import PyTree, PyTreeDef, Scalar

import time


class TrainingState:
    """
    Callback arguments object, designed to work in conjunction with :py:func:`klax.fit`.
    This class should not be instantiated directly.
    An instance of this class is passed to every callback object in the fit function.
    When writing a custom callback, use the properties of this class to access the
    current model, optimizer state, training data, and validation data during training.

    This class implements cached and lazy-evaluated values via property
    methods. This means that properties like training_loss are only calculated
    if they are used and are stored such that they are not calculated multiple
    times.
    """

    step: int  #: Current step-count of the training.
    time_on_last_update: float  #: Global time of the last :py:meth:`update` call.
    data: PyTree  #: PyTree of the training data.
    val_data: PyTree | None  #: PyTree of the validation data.
    _treedef_model: PyTreeDef
    _flat_model: list
    _treedef_opt_state: PyTreeDef
    _flat_opt_state: list
    _cache: dict = {}
    _get_loss: Callable[[PyTree, PyTree], Scalar]
    _start_time: float

    def __init__(
        self,
        get_loss: Callable[[PyTree, PyTree], Scalar],
        treedef_model: PyTreeDef,
        treedef_opt_state: PyTreeDef,
        data: PyTree,
        val_data: Optional[PyTree] = None,
    ):
        """Initializes the callback arguments object for one run of the fit :py:func:`klax.fit`.

        Args:
            get_loss: Function that takes a model and a batch of data and returns the loss.
            treedef_model: ``PyTreeDef`` of the model.
            treedef_opt_state: ``PyTreeDef`` of the :py:mod:`optax` optimizer.
            data: ``PyTree`` of the training data.
            val_data: ``PyTree`` of the validation data. If None, no validation loss is calculated and
                the property :py:attr:`val_loss` will return None.
        """
        self.data = data
        self.val_data = val_data
        self._get_loss = get_loss
        self._treedef_model = treedef_model
        self._treedef_opt_state = treedef_opt_state

    def update(self, flat_model: PyTree, flat_opt_state: PyTree, step: int):
        """Updates the callback arguments object with the current model and optimizer state.
        This method is called repeatedly in :py:func:`klax.fit`.

        Args:
            flat_model: Flattened ``PyTree`` of the model.
            flat_opt_state: Flattened ``PyTree`` of the :py:mod:`optax` optimizer.
            step: Current step-count of the training.
        """
        self._flat_model = flat_model
        self._flat_opt_state = flat_opt_state
        self.step = step
        self.time_on_last_update = time.time()

        # Clear cache
        self._cache = {}

    @staticmethod
    def _lazy_evaluated_and_cached(fun: Callable) -> property:
        """Turns a public method into a property

        The return value of `fun`is stored in the `_cache` dictionary of the
        current object using the function name as key. If the name is already in
        `_cache` then the cached value is simply returned, wihout evaluating
        `fun`.

        Args:
            fun: Method to wrap.

        Returns:
            Wraped method as a property.
        """
        attr_name = fun.__name__

        def wrapper(self):
            if attr_name not in self._cache:
                self._cache.setdefault(attr_name, fun(self))
            return self._cache.get(attr_name)

        wrapper.__doc__ = fun.__doc__

        return property(wrapper)

    @_lazy_evaluated_and_cached
    def model(self):
        """Lazy-evaluated and cached model."""
        return jax.tree_util.tree_unflatten(self._treedef_model, self._flat_model)

    @model.setter
    def model(self, model):
        self._flat_model, self._treedef_model = jax.tree.flatten(model)

    @_lazy_evaluated_and_cached
    def opt_state(self):
        """Lazy-evaluated and cached optimizer state."""
        return jax.tree_util.tree_unflatten(
            self._treedef_opt_state, self._flat_opt_state
        )

    @_lazy_evaluated_and_cached
    def loss(self):
        """Lazy-evaluated and cached training loss."""
        return self._get_loss(self.model, self.data)

    @_lazy_evaluated_and_cached
    def val_loss(self) -> Scalar | None:
        """Lazy-evaluated and cached validation loss."""
        if self.val_data is None:
            return None
        return self._get_loss(self.model, self.val_data)
