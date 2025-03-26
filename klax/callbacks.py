from collections.abc import Callable
import typing
from typing import Optional, Protocol

import jax
from jaxtyping import PyTree, PyTreeDef, Scalar

from .typing import DataTree


class CallbackArgs:
    """
    Callback arguments object, designed to work in conjunction with ``klax.fit``.

    It should not be used elsewhere!

    The instances of this class are passed to any callback object in the fit
    function. The class implements cached and lazy-evaluated values via property
    methods. This means that properties like training_loss are only calculated
    if they are used and are stored such that they are not calculated multiple
    times.
    """

    step: int
    data: DataTree
    val_data: DataTree | None
    _treedef_model: PyTreeDef
    _flat_model: list
    _cache: dict = {}
    _get_loss: Callable[..., Scalar]

    def __init__(
        self,
        get_loss: Callable[[PyTree, DataTree], Scalar],
        data: DataTree,
        val_data: Optional[DataTree],
        treedef_model: PyTreeDef,
    ):
        self.data = data
        self.val_data = val_data
        self._get_loss = get_loss  # lambda m: get_loss(m, *data)
        self._treedef_model = treedef_model

    def update(self, flat_model: PyTree, step: int):
        self._flat_model = flat_model
        self.step = step

        # Clear cache
        self._cache = {}

    def _lazy_evaluated_and_cached[T: Callable](fun: T) -> T:
        """Turns fun into property and stores method output in `_cache` `dict`
        of the class using the function name as key. If the fun name is already in
        `_cache` then the correspongind value is used instead of evaluating fun.

        **Arguments:**
            `fun`: Method to wrap.

        Returns:
            Wraped method.
        """
        attr_name = fun.__name__

        def new_fun(self):
            if attr_name not in self._cache:
                self._cache.setdefault(attr_name, fun(self))
            return self._cache.get(attr_name)

        return property(new_fun)

    @_lazy_evaluated_and_cached
    def model(self):
        return jax.tree_util.tree_unflatten(self._treedef_model, self._flat_model)

    @_lazy_evaluated_and_cached
    def loss(self):
        return self._get_loss(self.model, self.data)

    @_lazy_evaluated_and_cached
    def val_loss(self) -> Scalar | None:
        return self._get_loss(self.model, self.val_data)


@typing.runtime_checkable
class Callback(Protocol):
    """An abstract callback."""

    def __call__(self, cbargs: CallbackArgs) -> bool | None:
        raise NotImplementedError
