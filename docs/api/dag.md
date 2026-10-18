---
title: Graph-based Models
---

# Graph-based Models

Klax's DAG API lets you describe a computation as a directed acyclic graph
(DAG) of named variables and reusable blocks. A block consumes named inputs,
computes one or more outputs, and can call graph functions representing
subgraphs. A [`GraphModel`][klax.dag.GraphModel] pairs a graph with its
parameter PyTree, so the computation can be executed, differentiated, batched,
and compiled with JAX.

This is useful when a model is naturally a pipeline of distinct operations and
you want to be able to compute different outputs or intermediates in separate 
calls without redefining the model.

## Core Concepts

- **Block:** A named computation with declared input and output variable names
  and a callable that implements the computation.
- **Graph:** An immutable collection of blocks. It validates the graph, plans
  which blocks are needed for requested outputs, and executes that plan.
- **GraphModel:** An Equinox module that pairs a parameter PyTree with a graph.
  It provides convenient execution with the parameters already bound.
- **GraphFunction:** A callable view of part of a graph, bound to a parameter
  PyTree. Blocks can use these to apply transformations to subgraphs.

Variable names connect blocks: the output name of one block can be used as an
input name by another. Variables that are not produced by any block are graph
inputs and must be supplied by the caller. The graph determines execution
order from these dependencies; the order in which blocks are added is not a
substitute for declaring the correct dependencies.

## Building a Graph

`GraphBuilder` collects blocks through its `block` decorator. Each decorated
function receives the parameter PyTree first, followed by the values of its
declared inputs. The function name is used as the block name unless `name` is
provided.

```python
from typing import NamedTuple

import jax.numpy as jnp

from klax.dag import GraphBuilder, GraphModel


class Params(NamedTuple):
	scale: jnp.ndarray
	offset: jnp.ndarray


builder = GraphBuilder()


@builder.block(inputs="x", outputs="scaled")
def scale_input(params, x):
	return params.scale * x


@builder.block(inputs="scaled", outputs="y")
def add_offset(params, scaled):
	return scaled + params.offset


graph = builder.build()
params = Params(scale=jnp.array(2.0), offset=jnp.array(1.0))
model = GraphModel(params=params, graph=graph)
```

The `inputs` and `outputs` arguments accept a single name or an iterable of
names. Empty inputs are allowed for blocks that do not depend on runtime
variables, such as blocks that compute a matrix from parameters alone. A
function with one output returns that value; a function with multiple outputs
returns a tuple in the same order as the declared output names.

The decorator returns the original Python function, so it can still be called
directly. It also appends the corresponding block to the builder. Call
`build()` after registering all blocks to create the graph.

### Graph Functions

A block can declare one or more subgraphs it uses with the `uses` argument.
Each specification describes the subgraph's inputs and outputs using the form
`"input_a, input_b -> output_a, output_b"`:

```python
@builder.block(inputs="x", outputs="energy")
def energy(params, x):
	return params.energy_model(x)


@builder.block(inputs="x", outputs="energy_gradient", uses="x -> energy")
def energy_gradient(params, energy_function, x):
	return jax.grad(energy_function)(x)
```

When a block declares one subgraph, its callable is passed as the second
argument after `params`. When it declares multiple subgraphs, the second
argument is a tuple of callables in declaration order. With no `uses`, the
block receives only `params` and its input values. A graph function accepts
its declared inputs as positional arguments, in the order written before
`->`.

The graph checks that declared subgraphs can be planned and rejects recursive
subgraph use. This makes dependencies explicit while allowing a block to use
JAX transformations such as `jax.grad` on a graph-defined function.

## Planning and Execution

`Graph.plan(have, want)` returns the indices of the blocks needed to compute
the requested variables, in execution order. `have` lists variables already
available; `want` lists variables to compute. Planning raises `ValueError` if
an input is missing or if a required block would overwrite a supplied value.
Graphs also reject cycles and duplicate output variable names.

`Graph.run(params, have, want)` executes the plan and returns a dictionary
containing the requested variables. For example:

```python
result = graph.run(
	params,
	have={"x": jnp.array(3.0)},
	want=("y",),
)
assert result["y"] == 7.0
```

`GraphModel` binds `params`, so callers do not need to pass them on every run:

```python
result = model.run(have={"x": jnp.array(3.0)}, want=("y",))
# {"y": Array(7., dtype=float32, ...)}
```

Only requested variables are returned. Requesting a variable that is already
in `have` returns the supplied value without executing a producing block.
This is useful for partial computations and for reusing intermediate values.

`GraphModel.function(defn)` creates a [`GraphFunction`][klax.dag.GraphFunction]
bound to the model's current parameters. `defn` is a
[`GraphFunctionDef`][klax.dag.GraphFunctionDef] that names the function's
inputs and outputs. Calling the resulting function returns the single output
directly, or a tuple when it has multiple outputs.

## JAX and Equinox

`GraphModel` is an `equinox.Module`. Its `params` field is part of the dynamic
PyTree, while its `graph` field is static. This is the central distinction for
JAX transformations:

- Array parameters remain ordinary dynamic values. They can be differentiated
  and updated without rebuilding the graph.
- The graph structure and block callables are static Python data. They are
  inspected while tracing to construct the computation, rather than passed as
  arrays to compiled code.

If the parameter PyTree contains Klax constraint wrappers, call
`klax.finalize(model)` to apply and unwrap them before inference. Finalization
returns an unconstrained PyTree, so keep the original model for further
training and finalize a separate copy after parameter updates.

Use `eqx.filter_jit` to compile a function that accepts a model and array
inputs:

```python
import equinox as eqx


@eqx.filter_jit
def predict(model, x):
	return model.run(have={"x": x}, want=("y",))["y"]


y = predict(model, jnp.array(3.0))
```

For repeated calls with the same PyTree structure and array shapes and dtypes,
JAX reuses the compiled trace. Updating parameter array values alone does not
require tracing the graph again. A new trace or compilation can still be
expected when input or parameter shapes/dtypes change, the PyTree structure
changes, static arguments change, or a different graph is used. As with other
JAX code, Python block functions execute during tracing; compiled calls run
the resulting device program.

The same structure works with Klax's training utilities. Define a custom loss
with [`klax.loss`][klax.loss], then use its `value_and_grad` method to compute
the loss and its gradient with respect to the model:

```python
import jax
import klax


@klax.loss
def regression_loss(model, batch, run_state):
	x, target = batch
	prediction = jax.vmap(
		lambda sample: model.run({"x": sample}, ("y",))["y"]
	)(x)
	return jnp.mean(jnp.square(prediction - target))


loss_value, model_gradients = regression_loss.value_and_grad(
	model, (x_batch, y_batch), run_state=None
)
```

## Visualization

`to_dot(graph, have=..., want=...)` returns a Graphviz DOT description of the
graph. Without `want`, all blocks are drawn as active. If `want` is provided,
the same planning logic as `Graph.plan` is used: blocks outside the execution
plan are shown as inactive. Subgraphs declared by `uses` are highlighted with
colors and links to the blocks that use them.

```python
from klax.dag import to_dot

dot_source = to_dot(graph, have=("x",), want=("y",))
```

The returned string can be rendered with Graphviz or included in a notebook
using a Graphviz integration.

## API Reference

::: klax.dag.GraphModel
	options:
		members:
			- run
			- function

::: klax.dag.Graph
	options:
		members:
			- plan
			- run
			- build_graph_function

::: klax.dag.GraphBuilder
	options:
		members:
			- __init__
			- add_block
			- block
			- build

::: klax.dag.Block
    options:
		members:
			- name
            - inputs
            - outputs
            - fn
            - uses

::: klax.dag.GraphFunctionDef
    options:
		members:
            - inputs
            - outputs

::: klax.dag.GraphFunction
    options:
		members:
            - __call__

::: klax.dag.block

::: klax.dag.to_dot
