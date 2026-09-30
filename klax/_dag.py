"""Implements the core elements of a graph-based model building approach."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx


@dataclass(frozen=True)
class Edge:
    """An edge in a graph.

    Attributes:
        src: ID(s) of the inputs node(s) of the present edge.
        dst: ID(s) of the output node(s) of the present edge.
        apply: The function to apply to mapping from `src` to `dst`.

    """

    src: str | tuple[str, ...]
    dst: str | tuple[str, ...]
    apply: Callable


@dataclass(frozen=True)
class Graph:
    """A directed acyclic graph consisting of `Edge`s."""

    edges: dict[str, Edge]

    def run(
        self,
        modules: dict[str, Any],
        inputs: dict[str, Any],
        path: list[str],
        keep: set[str] | None = None,
    ) -> dict:
        """Seed node values with `inputs`, traverse `path`, return node values.

        If `keep` is given, nodes are pruned from the working dict as soon as
        they are no longer needed as edge inputs (and are absent from `keep`),
        and only the requested keys are returned.  Passing ``keep=None``
        (default) preserves the original behaviour of returning every node.

        Args:
            modules: A dictionary of modules containing the components used on
                the edges of the graph.
            inputs: A dictionary of inputs. The keys match the inputs nodes of
                of the provided path.
            path: A path along with the graph shall be evaluated given the
                input dictionary.
            keep: Whether or which intermediate results to keep in the output
                dictionary after having traversed the graph. The default is
                `None` meaning that all intermediate values are collected.

        Returns:
            A dictionary containing the outputs of the path as well as all
            stored intermediate values.

        """
        nodes = dict(inputs)

        # Precompute the last step index at which each node is consumed as a
        # source, so we know when it is safe to evict it.
        last_used: dict[str, int] | None = None
        if keep is not None:
            last_used = {}
            for i, name in enumerate(path):
                e = self.edges[name]
                srcs = e.src if isinstance(e.src, tuple) else (e.src,)
                for s in srcs:
                    last_used[s] = i

        for i, name in enumerate(path):
            e = self.edges[name]
            x = (
                tuple(nodes[s] for s in e.src)
                if isinstance(e.src, tuple)
                else nodes[e.src]
            )
            out = e.apply(modules, x)
            if isinstance(e.dst, tuple):
                for d, o in zip(e.dst, out):
                    nodes[d] = o
            else:
                nodes[e.dst] = out

            if last_used is not None:
                for key in [
                    k
                    for k in nodes
                    if k not in keep and last_used.get(k, -1) <= i
                ]:
                    del nodes[key]

        if keep is not None:
            return {k: nodes[k] for k in keep if k in nodes}
        return nodes


class DAGModel(eqx.Module):
    """A model consisting of modules and a directed acyclic graph."""

    modules: dict[str, Any]
    graph: Graph = eqx.field(static=True)

    def run(
        self, inputs: dict, path: list[str], keep: set[str] | None = None
    ) -> dict:
        """Traverse the graph along a given path, given a set of inputs.

        Args:
            inputs: A dictionary of inputs. The keys match the inputs nodes of
                of the provided path.
            path: A path along with the graph shall be evaluated given the
                input dictionary.
            keep: Whether or which intermediate results to keep in the output
                dictionary after having traversed the graph. The default is
                `None` meaning that all intermediate values are collected.


        Returns:
            A dictionary containing the outputs of the path as well as all
            stored intermediate values.

        """
        return self.graph.run(self.modules, inputs, path, keep)


def diff_edge(dst, wrt, out, subpath, graph, diff: Callable):
    """Create an `Edge` that differentiates a subgraph.

    Args:
        dst: The output nodes of the differentiation.
        wrt: The nodes with respect to which the subgraph is differentiated.
        out: The output of the subgraph, which is differentiated with respect
            to `wrt`.
        subpath: The path along the subgraph is evaluated.
        graph: The sub-graph that is differentiated.
        diff: The gradient function to apply for the differentiation, e.g.,
            `jax.grad`, ...

    Returns:
        A new `Edge` mapping from `wrt` to `dst`.

    """

    def apply(m, x):
        return diff(lambda v: graph.run(m, {wrt: v}, subpath)[out])(x)

    return Edge(src=wrt, dst=dst, apply=apply)


if __name__ == "__main__":
    import jax.numpy as jnp

    # In this example, we are building a DAG model for the
    # f(x,y) = sqr(x^2 + exp(y)) function.

    # Place all atomic building blocks in a module. Usually the modules are
    # the components holding the parameters. For this example we simply use
    # scalar functions
    modules = {"sqrt": jnp.sqrt, "exp": jnp.exp}

    # Next, we create the edges, which connect the node. Here, every node and
    # every edge has a unique name. Note, edges can also map from multiple
    # source to multiple destinations
    edges = {
        "x->x**2": Edge("x", "x**2", apply=lambda _, x: x**2),
        "y->exp(y)": Edge("y", "exp(y)", apply=lambda m, x: m["exp"](x)),
        "sum&sqrt": Edge(
            ("x**2", "exp(y)"), "z", apply=lambda m, x: m["sqrt"](x[0] + x[1])
        ),
    }

    graph = Graph(edges)
    model = DAGModel(modules, graph)
    print(model)

    # Now, let's evaluate the model along a provided path and with a certain
    # input seed. Thereby the path, simply contains the names of the edges
    # along the graph shall be evaluated.
    path = ["x->x**2", "y->exp(y)", "sum&sqrt"]
    seed = {
        "x": jnp.array(2.0),
        "y": jnp.array(1.5),
    }
    out = model.run(seed, path)

    # As you can see, the final outputs contains all intermediate as well as
    # the final result.
    print(out)
