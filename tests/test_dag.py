import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from klax.dag import Block, Graph, GraphFunctionDef, GraphModel


def _graph():
    return Graph(
        (
            Block(
                name="double",
                inputs=("x",),
                outputs=("twice_x",),
                fn=lambda params, subfns, x: (2 * x,),
                uses=(),
            ),
            Block(
                name="increment",
                inputs=("twice_x",),
                outputs=("y",),
                fn=lambda params, subfns, twice_x: (twice_x + 1,),
                uses=(),
            ),
        )
    )


def test_graph_plan_run_and_function():
    graph = _graph()

    assert graph.plan(("x",), ("y",)) == (0, 1)
    assert graph.run({}, {"x": jnp.array(3.0)}, ("y",))["y"] == 7

    function = graph.build_graph_function(
        {}, GraphFunctionDef(inputs=("x",), outputs=("y",))
    )
    assert function(jnp.array(3.0)) == 7


def test_graph_plan_rejects_missing_and_overwritten_variables():
    graph = _graph()
    multi_output_graph = Graph(
        (
            Block(
                name="pair",
                inputs=("x",),
                outputs=("y", "z"),
                fn=lambda params, subfns, x: (x, x),
                uses=(),
            ),
        )
    )

    with pytest.raises(ValueError, match="missing"):
        graph.plan((), ("y",))
    with pytest.raises(ValueError, match="overwritten"):
        multi_output_graph.plan(("x", "z"), ("y",))


def test_graph_model_does_not_retrace_for_parameter_updates():
    trace_count = 0

    def scale(params, subfns, x):
        nonlocal trace_count
        trace_count += 1
        return (params * x,)

    graph = Graph((Block("scale", ("x",), ("y",), scale, uses=()),))
    model = GraphModel(params=jnp.array(2.0), graph=graph)
    run = jax.jit(lambda model, x: model.run({"x": x}, ("y",))["y"])

    assert run(model, jnp.array(3.0)) == 6
    assert trace_count == 1

    updated_model = eqx.tree_at(
        lambda model: model.params, model, jnp.array(4.0)
    )
    assert run(updated_model, jnp.array(3.0)) == 12
    assert trace_count == 1
