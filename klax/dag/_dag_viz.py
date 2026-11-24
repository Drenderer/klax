import html
from collections.abc import Iterable
from dataclasses import dataclass

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

_Palette = tuple[str, str]


@dataclass(frozen=True)
class _VizInfo:
    """Format-independent description of what to draw for a `Graph`."""

    # Block indices that `graph.plan(have, want)` executes, or None if all
    # blocks are active.
    active: set[int] | None
    # block index -> palettes of all subfunctions the block belongs to.
    subfunction_blocks: dict[int, list[_Palette]]
    # block index -> (use index, subfunction, palette) per subfunction it uses.
    block_uses: dict[int, list[tuple[int, GraphFunctionDef, _Palette]]]
    # variable -> palette of the first subfunction it is an input/output of.
    variable_colors: dict[str, _Palette]
    # (subfunction output, using block index, palette, use index).
    subfunction_links: list[tuple[str, int, _Palette, int]]
    # All variables in a stable drawing order.
    variable_names: list[str]

    def is_inactive(self, block_index: int) -> bool:
        return self.active is not None and block_index not in self.active


def _collect(
    graph: Graph,
    have: Iterable[str],
    want: Iterable[str] | str | None,
) -> _VizInfo:
    active = (
        None
        if want is None
        else set(graph.plan(_as_tuple(have), _as_tuple(want)))
    )

    subfunction_blocks: dict[int, list[_Palette]] = {}
    block_uses: dict[int, list[tuple[int, GraphFunctionDef, _Palette]]] = {}
    variable_colors: dict[str, _Palette] = {}
    subfunction_links: list[tuple[str, int, _Palette, int]] = []

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

    variable_names: list[str] = []
    for block in graph.blocks:
        for variable in (*block.inputs, *block.outputs):
            if variable not in variable_names:
                variable_names.append(variable)
    for variable in variable_colors:
        if variable not in variable_names:
            variable_names.append(variable)

    return _VizInfo(
        active=active,
        subfunction_blocks=subfunction_blocks,
        block_uses=block_uses,
        variable_colors=variable_colors,
        subfunction_links=subfunction_links,
        variable_names=variable_names,
    )


def _uses_text(use_index: int, subfunction: GraphFunctionDef) -> str:
    return (
        f"uses ƒ{use_index}: {', '.join(subfunction.inputs)} → "
        f"{', '.join(subfunction.outputs)}"
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
    info = _collect(graph, have, want)

    def quote(value: object) -> str:
        """Quote a string for use as a DOT id or plain attribute value."""
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

    def format_edge(source: str, destination: str, block_index: int) -> str:
        colors = info.subfunction_blocks.get(block_index, [])
        is_inactive = info.is_inactive(block_index)
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
    for variable in info.variable_names:
        attributes = ["shape=ellipse", f"label={quote(variable)}"]
        if variable not in graph.producers:
            attributes += ["style=filled", "fillcolor=lightgray"]
        if variable in info.variable_colors:
            attributes += [
                f"color={quote(info.variable_colors[variable][0])}",
                "penwidth=2.5",
            ]
        lines.append(f"  {quote('v:' + variable)} [{', '.join(attributes)}];")

    # ---- block nodes and their edges --------------------------------------
    for block_index, block in enumerate(graph.blocks):
        block_id = f"b{block_index}"
        is_inactive = info.is_inactive(block_index)

        rows = [f"<tr><td><b>{html.escape(block.name)}</b></td></tr>"]
        for use_index, subfunction, (_, fill) in info.block_uses.get(
            block_index, []
        ):
            text = _uses_text(use_index, subfunction)
            rows.append(
                f'<tr><td bgcolor="{fill}">{html.escape(text)}</td></tr>'
            )
        label = (
            '<<table border="0" cellborder="0" cellspacing="0" cellpadding="3">'
            + "".join(rows)
            + "</table>>"
        )

        attributes = ["shape=box", f"label={label}"]
        if block_index in info.subfunction_blocks:
            attributes += [
                f"color={quote(info.subfunction_blocks[block_index][0][0])}",
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
    for variable, block_index, (color, _), use_index in info.subfunction_links:
        lines.append(
            f"  {quote('v:' + variable)} -> {quote(f'b{block_index}')} "
            f"[color={quote(color)}, style=dashed, constraint=false, "
            f"label={quote(f'ƒ{use_index}')}, fontcolor={quote(color)}];"
        )

    lines.append("}")
    return "\n".join(lines)


# Characters that would break a quoted Mermaid label, mapped to Mermaid's
# entity codes. "#" comes first because the other codes start with it.
_MERMAID_ESCAPES = str.maketrans(
    {
        "#": "#35;",
        '"': "#quot;",
        "<": "#lt;",
        ">": "#gt;",
        "&": "#amp;",
    }
)


def _mermaid_text(value: str) -> str:
    """Escape text for use inside a quoted Mermaid label."""
    return value.translate(_MERMAID_ESCAPES)


def to_mermaid(
    graph: Graph,
    have: Iterable[str] = (),
    want: Iterable[str] | str | None = None,
) -> str:
    """Return a Mermaid flowchart description of a `Graph`, flowing top to bottom.

    The drawing conventions match [`to_dot`][klax.dag.to_dot]:
      * Variables are rounded (stadium) nodes. Variables that no block
        produces (the graph's free inputs) are filled gray.
      * Blocks are boxes.
      * Every subfunction a block uses gets its own color. The using block
        lists one row per subfunction ("uses ƒ0: x → H"). The blocks that make
        up the subfunction have a border in that color, their edges are drawn
        in that color, and the subfunction's input/output variables have a
        colored outline. A dashed edge of the same color, labeled "ƒi", runs
        from each subfunction output to the using block.
      * If `want` is given, blocks that `graph.plan(have, want)` would not
        execute are drawn dashed and gray.

    Mermaid has fewer styling options than Graphviz, so there are some
    differences from `to_dot`:
      * If a block belongs to several subfunctions, its edges use the color of
        the first one only.
      * The "uses ƒi" rows in a block are not filled with color.
      * The dashed "uses" edges take part in the layout and can bend the
        top-to-bottom flow, because Mermaid cannot exclude edges from it.

    The returned string is plain Mermaid source. To embed it in Markdown,
    wrap it in a ```` ```mermaid ```` code fence.

    Args:
        graph (Graph): The graph to draw.
        have (Iterable[str]): Names of the supplied variables (only used
            together with `want`).
        want (Iterable[str] | str | None): Names of the requested variables,
            or None to draw everything as active.

    Returns:
        str: The Mermaid source.

    """
    info = _collect(graph, have, want)

    # Generated ids, since variable names may not be valid Mermaid ids.
    variable_ids = {
        variable: f"v{index}"
        for index, variable in enumerate(info.variable_names)
    }

    lines = ["flowchart TB"]
    node_styles: list[str] = []
    free_nodes: list[str] = []
    inactive_nodes: list[str] = []
    # linkStyle refers to edges by their definition order, so every edge gets
    # an index. Edges with the same style are grouped into one linkStyle line.
    edge_count = 0
    edge_styles: dict[str, list[int]] = {}

    def add_edge(
        source: str,
        destination: str,
        *,
        dashed: bool,
        style: str | None,
        label: str | None = None,
    ) -> None:
        nonlocal edge_count
        arrow = "-.->" if dashed else "-->"
        if label is not None:
            arrow += f'|"{_mermaid_text(label)}"|'
        lines.append(f"  {source} {arrow} {destination}")
        if style is not None:
            edge_styles.setdefault(style, []).append(edge_count)
        edge_count += 1

    # ---- variable nodes ---------------------------------------------------
    for variable in info.variable_names:
        variable_id = variable_ids[variable]
        lines.append(f'  {variable_id}(["{_mermaid_text(variable)}"])')
        if variable not in graph.producers:
            free_nodes.append(variable_id)
        if variable in info.variable_colors:
            color = info.variable_colors[variable][0]
            node_styles.append(
                f"  style {variable_id} stroke:{color},stroke-width:2.5px"
            )

    # ---- block nodes and their edges --------------------------------------
    for block_index, block in enumerate(graph.blocks):
        block_id = f"b{block_index}"
        is_inactive = info.is_inactive(block_index)

        rows = [f"<b>{_mermaid_text(block.name)}</b>"]
        for use_index, subfunction, _ in info.block_uses.get(block_index, []):
            rows.append(_mermaid_text(_uses_text(use_index, subfunction)))
        lines.append(f'  {block_id}["{"<br/>".join(rows)}"]')

        colors = info.subfunction_blocks.get(block_index, [])
        if colors:
            node_styles.append(
                f"  style {block_id} stroke:{colors[0][0]},stroke-width:2.5px"
            )
            edge_style = f"stroke:{colors[0][0]},stroke-width:2px"
        elif is_inactive:
            edge_style = "stroke:gray"
        else:
            edge_style = None
        if is_inactive:
            inactive_nodes.append(block_id)

        for variable in block.inputs:
            add_edge(
                variable_ids[variable],
                block_id,
                dashed=is_inactive,
                style=edge_style,
            )
        for variable in block.outputs:
            add_edge(
                block_id,
                variable_ids[variable],
                dashed=is_inactive,
                style=edge_style,
            )

    # ---- "uses" links: subfunction output -> using block -------------------
    for variable, block_index, (color, _), use_index in info.subfunction_links:
        add_edge(
            variable_ids[variable],
            f"b{block_index}",
            dashed=True,
            style=f"stroke:{color},color:{color}",
            label=f"ƒ{use_index}",
        )

    # ---- styles -------------------------------------------------------------
    if free_nodes:
        lines.append("  classDef free fill:#d3d3d3")
        lines.append(f"  class {','.join(free_nodes)} free")
    if inactive_nodes:
        lines.append(
            "  classDef inactive stroke:gray,stroke-dasharray:5 5,color:gray"
        )
        lines.append(f"  class {','.join(inactive_nodes)} inactive")
    lines += node_styles
    for style, indices in edge_styles.items():
        lines.append(
            f"  linkStyle {','.join(str(index) for index in indices)} {style}"
        )

    return "\n".join(lines)
