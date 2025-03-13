from collections.abc import Callable
import typing
from typing import Optional, Protocol

import jax
from jaxtyping import PyTree, PyTreeDef, Scalar

from .typing import DataTree


class CallbackArgs:
    """
    Callback Argument Object, designed to work in conjunction with klax.fit. 

    It should not be used elsewhere!

    The instances of this class are passed to any callback object in the fit function.
    The class implements cached and lazy-evaluated values via property methods. This 
    means that properties like training_loss are only calculated if they are used and 
    are stored such that they are not calculated multiple times.
    """
    step: int
    _treedef_model: PyTreeDef
    _flat_model: list
    _model: PyTree | None = None
    _loss: Scalar | None = None
    _val_loss: Scalar | None = None
    _get_loss: Callable[..., Scalar]
    _get_val_loss: Callable[..., Scalar | None]

    def __init__(
        self,
        get_loss: Callable[[PyTree, DataTree, DataTree], Scalar],
        data: tuple[DataTree, DataTree],
        val_data: Optional[tuple[DataTree, DataTree]],
        treedef_model: PyTreeDef
    ):
        self._get_loss = lambda m: get_loss(m, *data)
        if val_data:
            self._get_val_loss = lambda m: get_loss(m, *val_data)
        else:
            self._get_val_loss = lambda _: None
        self._treedef_model = treedef_model

    def update(self, flat_model: PyTree, step: int):
        self._flat_model = flat_model
        self.step = step

        # Reset private properties for lazy evaluation
        self._model = None
        self._loss = None
        self._val_loss = None

    @property
    def model(self):
        # If the following statement is false, it means that the model has
        # already been unflattened, hence it will be returned without change
        if self._model is None:
            self._model = jax.tree_util.tree_unflatten(self._treedef_model,
                                                       self._flat_model)
        return self._model

    @property
    def loss(self):
        # If the following statement is false, it means that the loss has
        # already been computed since the last update, hence it will be returned
        # without change.
        if self._loss is None:
            self._loss = self._get_loss(self.model)
        return self._loss

    @property
    def val_loss(self) -> Scalar | None:
        # If the following statement is false, it means that the validation loss
        # has already been computed since the last update, hence it will be
        # returned without change.
        if self._val_loss is None:
            self._val_loss = self._get_val_loss(self.model)
        return self._val_loss
    

@typing.runtime_checkable
class Callback(Protocol):
    """An abstract callback."""
    def __call__(
        self,
        cbargs: CallbackArgs
    ) -> bool | None:
        raise NotImplementedError