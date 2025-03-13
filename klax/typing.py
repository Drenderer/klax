"""A collection of useful type aliases used within klax."""
import typing
from typing import Any, Generator, Protocol, Sequence, TypeAlias

from jaxtyping import ArrayLike, PRNGKeyArray, PyTree, Scalar


MaskTree: TypeAlias = PyTree[bool]
DataTree: TypeAlias = PyTree[ArrayLike | None]

BatchGenerator: TypeAlias = Generator[DataTree, None, None]


@typing.runtime_checkable
class Dataloader(Protocol):
    def __call__(
        self,
        data: DataTree,
        batch_size: int,
        batch_mask: MaskTree | None,
        *,
        key: PRNGKeyArray
    ) -> BatchGenerator:
        raise NotImplementedError


@typing.runtime_checkable
class Loss(Protocol):
    def __call__(
        self,
        model: PyTree,
        x: DataTree,
        y: DataTree,
        in_axes: int | None | Sequence[Any] = 0
    ) -> Scalar:
        raise NotImplementedError
