from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ._dag import Block, Graph, GraphFunctionDef, _as_tuple


# Frozen dataclass wrapper to ensure equality and hashability of the block's callable.
@dataclass(frozen=True)
class BlockCallableWrapper:
    """Wrap a callable to make it a `BlockCallable`."""

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
    name=None,
):
    def decorator(fn):
        return _make_block(name or fn.__name__, inputs, outputs, fn, uses)

    return decorator


class GraphBuilder:
    """A builder for a `Graph`."""

    def __init__(self):
        self._blocks: list[Block] = []

    def block(
        self,
        inputs: str | Iterable[str],
        outputs: str | Iterable[str],
        uses: str | Iterable[str] | None = None,
        name=None,
    ):
        def decorator(fn):
            self._blocks.append(
                _make_block(name or fn.__name__, inputs, outputs, fn, uses)
            )
            return fn

        return decorator

    def build(self) -> Graph:
        """Return the built `Graph`."""
        return Graph(tuple(self._blocks))
