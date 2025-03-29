"""A collection of useful type aliases used within klax."""

from __future__ import annotations
from typing import Generator, TypeAlias

from jaxtyping import ArrayLike, PyTree


MaskTree: TypeAlias = PyTree[bool]
DataTree: TypeAlias = PyTree[ArrayLike | None]

BatchGenerator: TypeAlias = Generator[DataTree, None, None]
