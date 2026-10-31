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


def to_dot(
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
        graph (Graph): The graph to draw.
        have (Iterable[str]): Names of the supplied variables (only used
            together with `want`).
        want (Iterable[str] | str | None): Names of the requested variables,
            or None to draw everything as active.

    Returns:
        str: The DOT source.

    """
    active = (
        None
        if want is None
        else set(graph.plan(_as_tuple(have), _as_tuple(want)))
    )

    # ---- collect subfunction information --------------------------------
    subfunction_blocks: dict[int, list[tuple[str, str]]] = {}
    block_uses: dict[
        int, list[tuple[int, GraphFunctionDef, tuple[str, str]]]
    ] = {}
    variable_colors: dict[str, tuple[str, str]] = {}
    subfunction_links: list[tuple[str, int, tuple[str, str], int]] = []

    for subfunction_index, ((block_index, use_index), plan) in enumerate(
        graph._subplans.items()
    ):
        palette = _PALETTE[subfunction_index % len(_PALETTE)]
        subfunction = graph.blocks[block_index].uses[use_index]
        block_uses.setdefault(block_index, []).append(
            (use_index, subfunction, palette)
        )
        for planned_block_index in plan:
            subfunction_blocks.setdefault(planned_block_index, []).append(
                palette
            )
        for variable in (*subfunction.inputs, *subfunction.outputs):
            variable_colors.setdefault(variable, palette)
        for variable in subfunction.outputs:
            subfunction_links.append(
                (variable, block_index, palette, use_index)
            )

    # ---- helpers ----------------------------------------------------------
    def quote(value: object) -> str:
        """Quote a string for use as a DOT id or plain attribute value."""
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

    def format_edge(source: str, destination: str, block_index: int) -> str:
        colors = subfunction_blocks.get(block_index, [])
        is_inactive = active is not None and block_index not in active
        attributes = []
        if colors:
            attributes += [
                f"color={quote(':'.join(color[0] for color in colors))}",
                "penwidth=2",
            ]
        elif is_inactive:
            attributes.append("color=gray")
        if is_inactive:
            attributes.append("style=dashed")
        suffix = f" [{', '.join(attributes)}]" if attributes else ""
        return f"  {quote(source)} -> {quote(destination)}{suffix};"

    lines = [
        "digraph G {",
        "  rankdir=TB;",
        "  nodesep=0.4;",
        "  ranksep=0.5;",
        '  node [fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=10];',
    ]

    # ---- variable nodes ---------------------------------------------------
    variable_names: list[str] = []
    for block in graph.blocks:
        for variable in (*block.inputs, *block.outputs):
            if variable not in variable_names:
                variable_names.append(variable)
    for variable in variable_colors:
        if variable not in variable_names:
            variable_names.append(variable)

    for variable in variable_names:
        attributes = ["shape=ellipse", f"label={quote(variable)}"]
        if variable not in graph.producers:
            attributes += ["style=filled", "fillcolor=lightgray"]
        if variable in variable_colors:
            attributes += [
                f"color={quote(variable_colors[variable][0])}",
                "penwidth=2.5",
            ]
        lines.append(f"  {quote('v:' + variable)} [{', '.join(attributes)}];")

    # ---- block nodes and their edges --------------------------------------
    for block_index, block in enumerate(graph.blocks):
        block_id = f"b{block_index}"
        is_inactive = active is not None and block_index not in active

        rows = [f"<tr><td><b>{html.escape(block.name)}</b></td></tr>"]
        for use_index, subfunction, (_, fill) in block_uses.get(
            block_index, []
        ):
            text = (
                f"uses ƒ{use_index}: {', '.join(subfunction.inputs)} → "
                f"{', '.join(subfunction.outputs)}"
            )
            rows.append(
                f'<tr><td bgcolor="{fill}">{html.escape(text)}</td></tr>'
            )
        label = (
            '<<table border="0" cellborder="0" cellspacing="0" cellpadding="3">'
            + "".join(rows)
            + "</table>>"
        )

        attributes = ["shape=box", f"label={label}"]
        if block_index in subfunction_blocks:
            attributes += [
                f"color={quote(subfunction_blocks[block_index][0][0])}",
                "penwidth=2.5",
            ]
        elif is_inactive:
            attributes.append("color=gray")
        if is_inactive:
            attributes += ["style=dashed", "fontcolor=gray"]
        lines.append(f"  {quote(block_id)} [{', '.join(attributes)}];")

        for variable in block.inputs:
            lines.append(format_edge("v:" + variable, block_id, block_index))
        for variable in block.outputs:
            lines.append(format_edge(block_id, "v:" + variable, block_index))

    # ---- "uses" links: subfunction output -> using block -------------------
    # constraint=false keeps these edges from distorting the top-to-bottom layout.
    for variable, block_index, (color, _), use_index in subfunction_links:
        lines.append(
            f"  {quote('v:' + variable)} -> {quote(f'b{block_index}')} "
            f"[color={quote(color)}, style=dashed, constraint=false, "
            f"label={quote(f'ƒ{use_index}')}, fontcolor={quote(color)}];"
        )

    lines.append("}")
    return "\n".join(lines)
