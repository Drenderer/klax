"""Implements the core elements of a graph-based model building approach."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from graphlib import CycleError, TopologicalSorter
from typing import Any

import equinox as eqx
from jaxtyping import PyTree


def _as_tuple(x) -> tuple:
    return (x,) if not isinstance(x, tuple) else x


@dataclass(frozen=True)
class Block:
    """A compute node in a graph.

    Attributes:
        name: Name of the computation. Only used for visualizations.
        inputs: Names of the value nodes the block consumes.
        outputs: Names of the value nodes the block produces.
        fn: Callable implementing the block.
        subgraph: Optional nested graph the block may run (e.g. a vector field).

    """

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    fn: Callable[[PyTree, tuple], tuple]
    subgraph: "Graph | None" = None

    @classmethod
    def make(
        cls,
        name: str,
        inputs: str | Iterable[str],
        outputs: str | Iterable[str],
        fn: Callable,
        subgraph: "Graph | None" = None,
    ) -> "Block":
        return cls(name, _as_tuple(inputs), _as_tuple(outputs), fn, subgraph)

    def __post_init__(self):
        if not self.outputs:
            raise ValueError(f"block {self.name!r} has no outputs")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError(f"block {self.name!r} has duplicate outputs")
        if not callable(self.fn):
            raise TypeError(f"block {self.name!r}: fn must be callable")

    def __call__(self, params, *args):
        """Wrap fn and return a tuple, even if it has only one output."""
        out = self.fn(params, *args)
        return (out,) if len(self.outputs) == 1 else out


@dataclass(frozen=True)
class Graph:
    """A directed acyclic bipartite defined by multiple `Block`s."""

    blocks: tuple[Block, ...]

    def __post_init__(self):
        self._rank  # validates duplicates (via producers) and cycles

    # cached property to stay hashable
    @cached_property
    def producers(self) -> dict[str, int]:
        producers = {}
        for i, block in enumerate(self.blocks):
            for variable in block.outputs:
                if variable in producers:
                    raise ValueError(f"duplicate output {variable!r}")
                producers[variable] = i
        return producers

    @cached_property
    def _rank(self) -> dict[int, int]:
        deps = {
            i: {self.producers[v] for v in b.inputs if v in self.producers}
            for i, b in enumerate(self.blocks)
        }
        try:
            order = tuple(TopologicalSorter(deps).static_order())
        except CycleError as e:
            cycle = " -> ".join(self.blocks[i].name for i in e.args[1])
            raise ValueError(f"cycle in graph: {cycle}") from e
        return {i: r for r, i in enumerate(order)}

    def plan(
        self, have: Iterable[str], want: Iterable[str]
    ) -> tuple[int, ...]:
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

    def run(
        self,
        params: dict[str, Any],
        have: dict[str, Any],
        want: Sequence[str],
    ) -> dict:
        env = dict(have)
        for b in self.plan(have.keys(), want):
            block = self.blocks[b]
            out = block(params, *(env[i] for i in block.inputs))
            for k, v in zip(block.outputs, out, strict=True):
                env[k] = v
        return {w: env[w] for w in want}

    def to_dot(
        self,
        have: Iterable[str] = (),
        want: Iterable[str] | None = None,
    ) -> str:
        """Return a Graphviz DOT description of the graph.

        If `want` is given, blocks that `plan(have, want)` would not execute
        are drawn dashed and gray (top level only).
        """
        active = None if want is None else set(self.plan(have, want))
        lines = [
            "digraph G {",
            "  rankdir=LR;",
            "  compound=true;",
            '  node [fontname="Helvetica"];',
        ]
        self._dot_body(lines, "", active, "  ")
        lines.append("}")
        return "\n".join(lines)

    def _dot_body(
        self,
        lines: list[str],
        prefix: str,
        active: set[int] | None,
        indent: str,
    ) -> None:
        def q(s: str) -> str:
            return str(s).replace("\\", "\\\\").replace('"', '\\"')

        declared: set[str] = set()

        def var(name: str) -> str:
            vid = f"{prefix}v:{name}"
            if vid not in declared:
                declared.add(vid)
                lines.append(
                    f'{indent}"{q(vid)}" [shape=ellipse, label="{q(name)}"];'
                )
            return vid

        for i, b in enumerate(self.blocks):
            bid = f"{prefix}b{i}"
            inactive = active is not None and i not in active
            style = (
                ", style=dashed, color=gray, fontcolor=gray"
                if inactive
                else ""
            )
            lines.append(
                f'{indent}"{q(bid)}" [shape=box, label="{q(b.name)}"{style}];'
            )
            for v in b.inputs:
                lines.append(f'{indent}"{q(var(v))}" -> "{q(bid)}";')
            for v in b.outputs:
                lines.append(f'{indent}"{q(bid)}" -> "{q(var(v))}";')

            if b.subgraph is not None:
                cid = f"cluster_{bid}"
                lines.append(f'{indent}subgraph "{q(cid)}" {{')
                lines.append(
                    f'{indent}  label="{q(b.name)} (subgraph)"; style=rounded;'
                )
                b.subgraph._dot_body(lines, f"{bid}/", None, indent + "  ")
                lines.append(f"{indent}}}")
                if (
                    b.subgraph.blocks
                ):  # anchor for the dotted link to the cluster
                    lines.append(
                        f'{indent}"{q(bid)}" -> "{q(bid)}/b0" '
                        f'[style=dotted, arrowhead=none, lhead="{q(cid)}"];'
                    )


class GraphModel(eqx.Module):
    """A model consisting of modules and a directed acyclic graph."""

    params: PyTree
    graph: Graph = eqx.field(static=True)

    def run(self, have: dict, want: Sequence[str]) -> dict:
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


if __name__ == "__main__":
    from typing import NamedTuple

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from jaxtyping import Array

    from klax.nn import MLP

    class Normalizer(eqx.Module):
        shift: Array
        scale: Array

        def forward(self, x):
            return (x - self.shift) / self.scale

        def inverse(self, x):
            return self.scale * x + self.shift

    class Params(NamedTuple):
        normalizer: Normalizer
        mlp: MLP

    normalizer = Normalizer(shift=jnp.array(1.0), scale=jnp.array(3.0))
    mlp = MLP(in_size=3, out_size=3, width_sizes=[16], key=jr.key(0))

    params = Params(normalizer, mlp)

    graph = Graph(
        (
            Block.make(
                "normalize x",
                "x",
                "x_normalized",
                fn=lambda m, x: m.normalizer.forward(x),
            ),
            Block.make(
                "concat",
                ("x", "u"),
                "z",
                fn=lambda m, x, u: jnp.concat([x, u]),
            ),
            Block.make(
                "mlp",
                "z",
                "y_normalized",
                fn=lambda m, x: m.mlp(x),
            ),
            Block.make(
                "denormalize y",
                "y_normalized",
                "y",
                fn=lambda m, y: m.normalizer.inverse(y[0]),
            ),
        )
    )
    model = GraphModel(params, graph)
    print(model)

    have: dict[str, Array] = {
        "x": jnp.array([1.0, 2.0]),
        "u": jnp.array([1.0]),
    }
    want = ("y", "x_normalized")

    @eqx.filter_jit
    def run(model, have, want):
        return model.run(have, want)

    jax.config.update("jax_log_compiles", True)
    out = run(model, have, want)
    out = run(model, have, want)

    print(out)

    dot = graph.to_dot()  # whole graph
    dot = graph.to_dot(have=("x", "u"), want=("y",))  # grey out unused blocks
    print(dot)
