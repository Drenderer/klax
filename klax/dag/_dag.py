"""Implements the core elements of a graph-based model building approach."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property
from graphlib import CycleError, TopologicalSorter
from typing import Any, Protocol

import equinox as eqx
import jax
from jaxtyping import PyTree


def _as_tuple(x) -> tuple:
    return (x,) if isinstance(x, str) else tuple(x)


@dataclass(frozen=True)
class GraphFunctionDef:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


class GraphFunction(eqx.Module):
    """A slice of a graph, bound to params, as a pure callable."""

    params: PyTree
    graph: "Graph" = eqx.field(static=True)
    defn: GraphFunctionDef = eqx.field(static=True)

    def __call__(self, *args):
        if len(args) != len(self.defn.inputs):
            raise TypeError(
                f"expected {len(self.defn.inputs)} arguments, got {len(args)}"
            )
        env = self.graph.run(
            self.params,
            have=dict(zip(self.defn.inputs, args, strict=True)),
            want=self.defn.outputs,
        )
        out = tuple(env[o] for o in self.defn.outputs)
        return out[0] if len(out) == 1 else out


class BlockCallable(Protocol):
    """A callable that can be used as a block in a graph."""

    def __call__(
        self, params: PyTree, subfns: tuple[GraphFunction, ...], *args
    ) -> Any: ...


@dataclass(frozen=True)
class Block:
    """A compute node in a graph.

    Attributes:
        name: Name of the computation. Only used for visualizations.
        inputs: Names of the value nodes the block consumes.
        outputs: Names of the value nodes the block produces.
        fn: Callable implementing the block. It takes the parameters, the
            required sub-graphs and the inputs, and computes the outputs.
        uses: A tuple of sub-graphDefs, that define the sub-graphs used
            by this Block.

    """

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    fn: BlockCallable
    uses: tuple[GraphFunctionDef, ...] = ()

    def __post_init__(self):
        if not self.outputs:
            raise ValueError(f"block {self.name!r} has no outputs")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError(f"block {self.name!r} has duplicate outputs")
        if not callable(self.fn):
            raise TypeError(f"block {self.name!r}: fn must be callable")


@dataclass(frozen=True)
class Graph:
    """A directed acyclic bipartite graph defined by multiple `Block`s."""

    blocks: tuple[Block, ...]

    def __post_init__(self):
        self._check_valid()
        self._hash

    @cached_property
    def _hash(self) -> int:
        return hash(self.blocks)

    def __hash__(self) -> int:
        return self._hash

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

    @cached_property
    def _plan_cache(self) -> dict:
        return {}

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
        have, want = frozenset(_as_tuple(have)), _as_tuple(want)
        try:
            return self._plan_cache[have, want]
        except KeyError:
            result = self._plan(have, want)
            self._plan_cache[have, want] = result
            return result

    def _plan(self, have: frozenset[str], want: tuple[str]) -> tuple[int, ...]:
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
        """Compute the plan for the j-th sub-graph of the i-th block."""
        plans = {}
        for i, b in enumerate(self.blocks):
            for j, sf in enumerate(b.uses):
                try:
                    plans[i, j] = self.plan(sf.inputs, sf.outputs)
                except ValueError as e:
                    raise ValueError(
                        f"block {b.name!r}, sub-graph {j}: {e}"
                    ) from e
        return plans

    def _check_valid(self):
        # First, let's check that the overall graph is acyclic
        self._rank

        # Next, we compute the call graph. In this directed graph each node is a
        # block and an edge A -> B indicates that B uses A in at least one of
        # it's sub-graphs.
        calls = {
            i: {k for j in range(len(b.uses)) for k in self._subplans[i, j]}
            for i, b in enumerate(self.blocks)
        }
        # We now have to check if the call graph is acyclic. If it is not, then
        # there exists at least one block who's sub-graph requires the
        # exection of itself, which is not allowed.
        try:
            tuple(TopologicalSorter(calls).static_order())
        except CycleError as e:
            cycle = " -> ".join(self.blocks[i].name for i in e.args[1])
            raise ValueError(f"recursive sub-graph use: {cycle}") from e

    def build_graph_function(
        self, params: PyTree, defn: GraphFunctionDef
    ) -> GraphFunction:
        """Build a GraphFunction bound to the given parameters."""
        self.plan(defn.inputs, defn.outputs)  # Check
        return GraphFunction(params, self, defn)

    def run(
        self,
        params: PyTree,
        have: dict[str, Any],
        want: tuple[str, ...],
    ) -> dict:
        env = dict(have)
        want = _as_tuple(want)
        for b in self.plan(tuple(have.keys()), want):
            block = self.blocks[b]
            subfns = tuple(GraphFunction(params, self, d) for d in block.uses)
            out = block.fn(params, subfns, *(env[i] for i in block.inputs))
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
            A dictionary containing the variables in want.

        """
        return self.graph.run(self.params, have, want)

    def function(self, defn: GraphFunctionDef) -> GraphFunction:
        """Return a GraphFunction bound to the model's parameters."""
        return self.graph.build_graph_function(self.params, defn)
