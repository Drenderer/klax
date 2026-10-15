import html
from collections.abc import Iterable

from ._dag import Graph, GraphFunctionDef, _as_tuple

# (edge/border color, light fill color) pairs, one per subfunction.
_PALETTE = (
    ("#d62728", "#f8d4d4"),  # red
    ("#1f77b4", "#d3e5f4"),  # blue
    ("#2ca02c", "#d6efd6"),  # green
    ("#ff7f0e", "#fde3c8"),  # orange
    ("#9467bd", "#e6dcf0"),  # purple
    ("#17becf", "#d0f0f4"),  # cyan
)


def graph_to_dot(
    graph: Graph,
    have: Iterable[str] = (),
    want: Iterable[str] | str | None = None,
) -> str:
    """Return a Graphviz DOT description of a `Graph`, flowing top to bottom.

    Drawing conventions:
      * Variables are ellipses. Variables that no block produces (the graph's
        free inputs) are filled gray.
      * Blocks are boxes.
      * Every subfunction a block uses gets its own color. The using block
        shows one colored row per subfunction ("uses ƒ0: x -> H"). The blocks
        that make up the subfunction have a border in that color, their edges
        are drawn in that color (several colors are drawn side by side if a
        block belongs to several subfunctions), and the subfunction's
        input/output variables have a colored outline. A dashed edge of the
        same color runs from each subfunction output to the using block.
      * If `want` is given, blocks that `graph.plan(have, want)` would not
        execute are drawn dashed and gray.

    Args:
        graph: The graph to draw.
        have: Names of the supplied variables (only used together with `want`).
        want: Names of the requested variables, or None to draw everything
            as active.

    Returns:
        The DOT source as a string.

    """
    active = (
        None
        if want is None
        else set(graph.plan(_as_tuple(have), _as_tuple(want)))
    )

    # ---- collect subfunction information --------------------------------
    island: dict[int, list[tuple[str, str]]] = {}  # block -> palette entries
    uses_rows: dict[
        int, list[tuple[int, GraphFunctionDef, tuple[str, str]]]
    ] = {}
    var_marks: dict[str, tuple[str, str]] = {}
    dashed_links: list[tuple[str, int, tuple[str, str], int]] = []

    for k, ((i, j), plan) in enumerate(graph._subplans.items()):
        pal = _PALETTE[k % len(_PALETTE)]
        sf = graph.blocks[i].uses[j]
        uses_rows.setdefault(i, []).append((j, sf, pal))
        for b in plan:
            island.setdefault(b, []).append(pal)
        for v in (*sf.inputs, *sf.outputs):
            var_marks.setdefault(v, pal)
        for v in sf.outputs:
            dashed_links.append((v, i, pal, j))

    # ---- helpers ----------------------------------------------------------
    def q(s: object) -> str:
        """Quote a string for use as a DOT id or plain attribute value."""
        return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'

    def edge(src: str, dst: str, block: int) -> str:
        pals = island.get(block, [])
        inactive = active is not None and block not in active
        attrs = []
        if pals:
            attrs += [f"color={q(':'.join(p[0] for p in pals))}", "penwidth=2"]
        elif inactive:
            attrs.append("color=gray")
        if inactive:
            attrs.append("style=dashed")
        suffix = f" [{', '.join(attrs)}]" if attrs else ""
        return f"  {q(src)} -> {q(dst)}{suffix};"

    lines = [
        "digraph G {",
        "  rankdir=TB;",
        "  nodesep=0.4;",
        "  ranksep=0.5;",
        '  node [fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=10];',
    ]

    # ---- variable nodes ---------------------------------------------------
    names: list[str] = []
    for b in graph.blocks:
        for v in (*b.inputs, *b.outputs):
            if v not in names:
                names.append(v)
    for sf_vars in var_marks:
        if sf_vars not in names:
            names.append(sf_vars)

    for v in names:
        attrs = ["shape=ellipse", f"label={q(v)}"]
        if v not in graph.producers:
            attrs += ["style=filled", "fillcolor=lightgray"]
        if v in var_marks:
            attrs += [f"color={q(var_marks[v][0])}", "penwidth=2.5"]
        lines.append(f"  {q('v:' + v)} [{', '.join(attrs)}];")

    # ---- block nodes and their edges --------------------------------------
    for i, b in enumerate(graph.blocks):
        bid = f"b{i}"
        inactive = active is not None and i not in active

        rows = [f"<tr><td><b>{html.escape(b.name)}</b></td></tr>"]
        for j, sf, (_, fill) in uses_rows.get(i, []):
            text = (
                f"uses ƒ{j}: {', '.join(sf.inputs)} → {', '.join(sf.outputs)}"
            )
            rows.append(
                f'<tr><td bgcolor="{fill}">{html.escape(text)}</td></tr>'
            )
        label = (
            '<<table border="0" cellborder="0" cellspacing="0" cellpadding="3">'
            + "".join(rows)
            + "</table>>"
        )

        attrs = ["shape=box", f"label={label}"]
        if i in island:
            attrs += [f"color={q(island[i][0][0])}", "penwidth=2.5"]
        elif inactive:
            attrs.append("color=gray")
        if inactive:
            attrs += ["style=dashed", "fontcolor=gray"]
        lines.append(f"  {q(bid)} [{', '.join(attrs)}];")

        for v in b.inputs:
            lines.append(edge("v:" + v, bid, i))
        for v in b.outputs:
            lines.append(edge(bid, "v:" + v, i))

    # ---- "uses" links: subfunction output -> using block -------------------
    # constraint=false keeps these edges from distorting the top-to-bottom layout.
    for v, i, (color, _), j in dashed_links:
        lines.append(
            f"  {q('v:' + v)} -> {q(f'b{i}')} "
            f"[color={q(color)}, style=dashed, constraint=false, "
            f"label={q(f'ƒ{j}')}, fontcolor={q(color)}];"
        )

    lines.append("}")
    return "\n".join(lines)
