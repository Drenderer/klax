"""Implements the core elements of a graph-based model building approach."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache
from graphlib import CycleError, TopologicalSorter
from typing import Any

import equinox as eqx
from jaxtyping import PyTree


@dataclass(frozen=True)
class SubFunctionDef:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


def _as_tuple(x) -> tuple:
    return (x,) if isinstance(x, str) else tuple(x)


@dataclass(frozen=True)
class Block:
    """A compute node in a graph.

    Attributes:
        name: Name of the computation. Only used for visualizations.
        inputs: Names of the value nodes the block consumes.
        outputs: Names of the value nodes the block produces.
        fn: Callable implementing the block. It takes the parameters, the
            required subfunctions and the inputs, and computes the outputs.
        uses: A tuple of SubFunctionDefs, that define the subfunctions used
            by this Block.

    """

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    fn: Callable[
        [PyTree, tuple[Callable, ...], *tuple[Any, ...]],
        tuple[Any, ...],
    ]
    uses: tuple[SubFunctionDef, ...] = ()

    @classmethod
    def make(
        cls,
        name: str,
        inputs: str | Iterable[str],
        outputs: str | Iterable[str],
        fn: Callable,
        uses: tuple[SubFunctionDef, ...] = (),
    ) -> "Block":
        return cls(name, _as_tuple(inputs), _as_tuple(outputs), fn, uses)

    def __post_init__(self):
        if not self.outputs:
            raise ValueError(f"block {self.name!r} has no outputs")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError(f"block {self.name!r} has duplicate outputs")
        if not callable(self.fn):
            raise TypeError(f"block {self.name!r}: fn must be callable")

    def __call__(self, params, subfns, *args):
        """Wrap fn and return a tuple, even if it has only one output."""
        out = self.fn(params, subfns, *args)
        return (out,) if len(self.outputs) == 1 else out


@dataclass(frozen=True)
class Graph:
    """A directed acyclic bipartite graph defined by multiple `Block`s."""

    blocks: tuple[Block, ...]

    def __post_init__(self):
        self._check_valid()

    # cached property to stay hashable
    @cached_property
    def producers(self) -> dict[str, int]:
        """Computes the mapping from output variable names to the index of the block that produces them."""
        producers = {}
        for i, block in enumerate(self.blocks):
            for variable in block.outputs:
                if variable in producers:
                    raise ValueError(f"Duplicate output {variable!r}")
                producers[variable] = i
        return producers

    @cached_property
    def _rank(self) -> dict[int, int]:
        """Compute the mapping from block indices to their topological rank (the order in which they should be executed)."""
        # Build the dependency graph: A directed graph, where each node is a
        # block and an edge A -> B indicates, that at least one of block A's
        # outputs is used by B.
        deps = {
            i: {self.producers[v] for v in b.inputs if v in self.producers}
            for i, b in enumerate(self.blocks)
        }
        # Next, check that the dependency graph is acyclic.
        try:
            order = tuple(TopologicalSorter(deps).static_order())
        except CycleError as e:
            cycle = " -> ".join(self.blocks[i].name for i in e.args[1])
            raise ValueError(f"cycle in graph: {cycle}") from e
        return {i: r for r, i in enumerate(order)}

    # TODO: The plans could be cached, to not recompute them every time.
    def plan(
        self, have: Iterable[str], want: Iterable[str]
    ) -> tuple[int, ...]:
        """Return topologically sorted block incices to compute `want` from `have`.

        Args:
            have (Iterable[str]): Iterable of variable names that are available.
            want (Iterable[str]): Iterable of variable names that should be
                computed.

        Raises:
            ValueError: If `want` can't be computed from `have`.
            ValueError: If an intermediate variable must be computed, but is
                already in have.

        Returns:
            tuple[int, ...]: Tuple of block indices in the order in which they
                should be executed.

        """
        want, have = tuple(want), frozenset(have)
        required_blocks: set[int] = set()
        seen_variables: set[str] = set()
        missing: set[str] = set()
        stack = list(want)
        while stack:
            v = stack.pop()
            if v in have or v in seen_variables:
                continue
            seen_variables.add(v)
            b = self.producers.get(v)
            if b is None:
                missing.add(v)
                continue
            required_blocks.add(b)
            stack.extend(self.blocks[b].inputs)

        if missing:
            raise ValueError(
                f"cannot compute {sorted(want)}: missing {sorted(missing)}"
            )

        clash = {
            v: self.blocks[b].name
            for b in required_blocks
            for v in self.blocks[b].outputs
            if v in have
        }
        if clash:
            raise ValueError(
                f"supplied values would be overwritten by blocks: {clash}"
            )

        return tuple(sorted(required_blocks, key=self._rank.__getitem__))

    @cached_property
    def _subplans(self) -> dict[tuple[int, int], tuple[int, ...]]:
        """Compute the plan for the j-th subfunction of the i-th block."""
        plans = {}
        for i, b in enumerate(self.blocks):
            for j, sf in enumerate(b.uses):
                try:
                    plans[i, j] = self.plan(sf.inputs, sf.outputs)
                except ValueError as e:
                    raise ValueError(
                        f"block {b.name!r}, subfunction {j}: {e}"
                    ) from e
        return plans

    def _check_valid(self):
        # First, let's check that the overall graph is acyclic
        self._rank

        # Next, we compute the call graph. In this directed graph each node is a
        # block and an edge A -> B indicates that B uses A in at least one of
        # it's subfunctions.
        calls = {
            i: {k for j in range(len(b.uses)) for k in self._subplans[i, j]}
            for i, b in enumerate(self.blocks)
        }
        # We now have to check if the call graph is acyclic. If it is not, then
        # there exists at least one block who's subfunction requires the
        # exection of itself, which is not allowed.
        try:
            tuple(TopologicalSorter(calls).static_order())
        except CycleError as e:
            cycle = " -> ".join(self.blocks[i].name for i in e.args[1])
            raise ValueError(f"recursive subfunction use: {cycle}") from e

    def build_subfunction(self, subfndef: SubFunctionDef) -> Callable:
        """Build a callable `subfn: (params, *inputs) -> (*outputs)`.

        Args:
            subfndef (SubFunctionDef): Definition of the subfunction.

        Returns:
            Callable: Callable subfunction.

        """

        def subfn(params, *args):
            env = self.run(
                params,
                have=dict(zip(subfndef.inputs, args, strict=True)),
                want=subfndef.outputs,
            )
            return tuple(env[o] for o in subfndef.outputs)

        return subfn

    def run(
        self,
        params: dict[str, Any],
        have: dict[str, Any],
        want: tuple[str, ...],
    ) -> dict:
        env = dict(have)
        for b in self.plan(tuple(have.keys()), want):
            block = self.blocks[b]
            subfns = tuple(
                self.build_subfunction(subfndef) for subfndef in block.uses
            )
            out = block(params, subfns, *(env[i] for i in block.inputs))
            for k, v in zip(block.outputs, out, strict=True):
                env[k] = v
        return {w: env[w] for w in want}


class GraphModel(eqx.Module):
    """A model consisting of modules and a directed acyclic graph."""

    params: PyTree
    graph: Graph = eqx.field(static=True)

    def run(self, have: dict, want: tuple[str, ...]) -> dict:
        """Compute the variables in want.

        Args:
            have: A dictionary of have, mapping each variable name to the
                corresponding input object.
            want: A sequence of variable names to compute.

        Returns:
            A dictionary containing the outputs of the path as well as all
            stored intermediate values.

        """
        return self.graph.run(self.params, have, want)
