from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ._dag import Block, Graph, GraphFunctionDef, _as_tuple


# Frozen dataclass wrapper to ensure equality and hashability of the block's callable.
@dataclass(frozen=True)
class BlockCallableWrapper:
    """Convenience wrapper for turning a callable into the required format for a [Block][klax.dag.Block]."""

    fn: Callable
    len_uses: int
    singelton_output: bool

    def __call__(self, params, subfns, *args):
        if self.len_uses == 0:
            out = self.fn(params, *args)
        elif self.len_uses == 1:
            out = self.fn(params, subfns[0], *args)
        else:
            out = self.fn(params, subfns, *args)
        return (out,) if self.singelton_output else out


def _gfd(spec: str) -> GraphFunctionDef:
    """Parse "x, y -> f, g" into GraphFunctionDef(("x", "y"), ("f", "g"))."""
    try:
        lhs, rhs = spec.split("->")
    except ValueError:
        raise ValueError(
            f"expected 'inputs -> outputs', got {spec!r}"
        ) from None

    def names(side: str) -> tuple[str, ...]:
        return tuple(s.strip() for s in side.split(",") if s.strip())

    return GraphFunctionDef(names(lhs), names(rhs))


def _make_block(
    name: str,
    inputs: str | Iterable[str],
    outputs: str | Iterable[str],
    fn: Callable,
    uses: str | Iterable[str] | None = None,
) -> Block:
    """Create a `Block` from a callable and its input/output names."""
    inputs = _as_tuple(inputs)
    outputs = _as_tuple(outputs)

    uses: tuple[GraphFunctionDef, ...] = (
        () if uses is None else tuple(_gfd(u) for u in _as_tuple(uses))
    )

    fn_wrapped = BlockCallableWrapper(fn, len(uses), len(outputs) == 1)

    return Block(
        name=name, inputs=inputs, outputs=outputs, fn=fn_wrapped, uses=uses
    )


def block(
    inputs: str | Iterable[str],
    outputs: str | Iterable[str],
    uses: str | Iterable[str] | None = None,
    name: str | None = None,
):
    """Decorate a function to turn it into a [`Block`][klax.dag.Block].

    Args:
        inputs (str | Iterable[str]): Input variable name(s).
        outputs (str | Iterable[str]): Output variable name(s).
        uses (str | Iterable[str] | None, optional): Graph function spec(s)
            with format 'a, b, c -> x, y'. Defaults to None.
        name (str, optional): Optional name for the block. Defaults to None,
            which uses the function's name.

    """

    def decorator(fn):
        return _make_block(name or fn.__name__, inputs, outputs, fn, uses)

    return decorator


class GraphBuilder:
    """Convenience class for building a [`Graph`][klax.dag.Graph] from [`Block`][klax.dag.Block]s."""

    def __init__(self):
        self._blocks: list[Block] = []

    def add_block(self, block: Block):
        """Add a [`Block`][klax.dag.Block] to the graph.

        Args:
            block (Block): The block to add.

        """
        self._blocks.append(block)

    def block(
        self,
        inputs: str | Iterable[str],
        outputs: str | Iterable[str],
        uses: str | Iterable[str] | None = None,
        name: str | None = None,
    ):
        """Return a function that creates a [`Block`][klax.dag.Block] and adds it to the graph.

        This method is inteded to be used as a decorator.
        !!! Example
            ```python
            g = GraphBuilder()


            @g.block(inputs="x", outputs="y")
            def my_block(params, x):
                return x + 1
            ```
        !!!Note:
            The decorated function is not modified, and can still be called
            directly.

        Args:
            inputs (str | Iterable[str]): Input variable name(s).
            outputs (str | Iterable[str]): Output variable name(s).
            uses (str | Iterable[str] | None, optional): Graph function spec(s)
                with format 'a, b, c -> x, y'. Defaults to None.
            name (str, optional): Optional name for the block. Defaults to None,
                which uses the function's name.

        """

        def decorator(fn):
            self._blocks.append(
                _make_block(name or fn.__name__, inputs, outputs, fn, uses)
            )
            return fn

        return decorator

    def build(self) -> Graph:
        """Build the graph.

        Returns:
            Graph: Graph consisting of the added blocks.

        """
        return Graph(tuple(self._blocks))
