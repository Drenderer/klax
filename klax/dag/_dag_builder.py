import inspect
import re
from collections.abc import Callable, Iterable

from ._dag import Block, Graph, SubFunctionDef, _as_tuple

_SPEC = re.compile(r"^\s*([^()]+?)\s*\(\s*([^()]*?)\s*\)\s*$")


def _split(s: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in s.split(",") if p.strip())


def parse_subfunction(spec: str) -> SubFunctionDef:
    """Parse "out1, out2(in1, in2)" into a SubFunctionDef."""
    m = _SPEC.match(spec)
    if m is None:
        raise ValueError(
            f"bad subfunction spec {spec!r}; expected 'out(in1, in2)'"
        )
    return SubFunctionDef(inputs=_split(m[2]), outputs=_split(m[1]))


class _Sub:
    """A subfunction bound to params. Call it as f(*inputs)."""

    def __init__(self, raw: Callable, params, single: bool):
        self._raw, self._params, self._single = raw, params, single

    def unbound(self, params, *args):
        out = self._raw(params, *args)
        return out[0] if self._single else out

    def __call__(self, *args):
        return self.unbound(self._params, *args)


class GraphBuilder:
    def __init__(self):
        self._blocks: list[Block] = []

    def block(
        self,
        outputs: str | Iterable[str] | None = None,
        uses: str | Iterable[str] = (),
        inputs: str | Iterable[str] | None = None,
        name: str | None = None,
    ):
        specs = tuple(
            parse_subfunction(s)
            for s in ((uses,) if isinstance(uses, str) else uses)
        )

        def deco(fn):
            params = list(inspect.signature(fn).parameters)
            n_skip = 1 + len(
                specs
            )  # params tree + one parameter per subfunction
            if len(params) < n_skip:
                raise TypeError(
                    f"{fn.__name__}: needs (params, {len(specs)} subfunction(s), *inputs)"
                )
            ins = (
                _as_tuple(inputs)
                if inputs is not None
                else tuple(params[n_skip:])
            )
            outs = (
                _as_tuple(outputs) if outputs is not None else (fn.__name__,)
            )
            singles = tuple(len(sf.outputs) == 1 for sf in specs)

            def wrapped(p, raw_subfns, *args):
                subs = tuple(
                    _Sub(r, p, s)
                    for r, s in zip(raw_subfns, singles, strict=True)
                )
                return fn(p, *subs, *args)

            self._blocks.append(
                Block(name or fn.__name__, ins, outs, wrapped, specs)
            )
            return (
                fn  # the decorated function stays a plain, testable function
            )

        return deco

    def build(self) -> Graph:
        return Graph(tuple(self._blocks))
