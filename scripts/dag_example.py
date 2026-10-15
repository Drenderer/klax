from typing import NamedTuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

import klax
from klax.dag import GraphBuilder, GraphModel, graph_to_dot
from klax.nn import FICNN


class Normalizer(eqx.Module):
    shift: Array
    scale: Array

    def forward(self, x):
        return (x - self.shift) / self.scale

    def inverse(self, x):
        return self.scale * x + self.shift


class Params(NamedTuple):
    ts_normalizer: Normalizer
    ys_normalizer: Normalizer
    us_normalizer: Normalizer
    ficnn: FICNN
    l_matrix: Array
    a_matrix: Array
    input_matrix: Array
    epsilon: Array
    delta: Array
    x_star: Array


def make_params(
    key, input_size: int, state_size: int, output_size: int
) -> Params:
    key, subkey = jr.split(key)
    ficnn = FICNN(
        in_size=state_size,
        out_size="scalar",
        width_sizes=(16, 16),
        key=subkey,
    )

    key, subkey = jr.split(key)
    l_matrix = jr.normal(subkey, (state_size, state_size))

    key, subkey = jr.split(key)
    a_matrix = jr.normal(subkey, (state_size, state_size))

    key, subkey = jr.split(key)
    input_matrix = jr.normal(subkey, (state_size, input_size))

    epsilon = jnp.array(0.1)
    delta = jnp.array(0.1)
    x_star = jnp.zeros(state_size)
    ts_normalizer = Normalizer(shift=jnp.array(0.0), scale=jnp.array(1.0))
    ys_normalizer = Normalizer(
        shift=jnp.zeros(output_size), scale=jnp.ones(output_size)
    )
    us_normalizer = Normalizer(
        shift=jnp.zeros(input_size), scale=jnp.ones(input_size)
    )
    return Params(
        ts_normalizer=ts_normalizer,
        ys_normalizer=ys_normalizer,
        us_normalizer=us_normalizer,
        ficnn=ficnn,
        l_matrix=l_matrix,
        a_matrix=a_matrix,
        input_matrix=input_matrix,
        epsilon=epsilon,
        delta=delta,
        x_star=x_star,
    )


if __name__ == "__main__":
    input_size, state_size, output_size = 2, 6, 3
    key = jr.PRNGKey(0)
    params = make_params(key, input_size, state_size, output_size)
    g = GraphBuilder()

    # --- Hamiltonian ---------------------------------------------------------
    @g.block(inputs="x", outputs="f")
    def icnn(params, x):
        return params.ficnn(x)

    @g.block(inputs="x", outputs="f_bar", uses="x -> f")
    def bregman_divergence(params, f, x):
        x_star = params.x_star
        return f(x) - f(x_star) - jnp.dot(jax.grad(f)(x_star), (x - x_star))

    @g.block(inputs=("f_bar", "x"), outputs="hamiltonian")
    def add_quadratic(params, f_bar, x):
        dx = x - params.x_star
        return f_bar + 0.5 * jnp.inner(dx, dx)

    @g.block(
        inputs="x", outputs="gradient_hamiltonian", uses="x -> hamiltonian"
    )
    def gradient_hamiltonian(params, hamiltonian, x):
        return jax.grad(hamiltonian)(x)

    # --- system matrices -----------------------------------------------------
    @g.block(inputs=(), outputs="structure_matrix")
    def structure_matrix(params):
        return params.a_matrix - params.a_matrix.T

    @g.block(inputs=(), outputs="dissipation_matrix")
    def dissipation_matrix(params):
        return (
            params.epsilon * jnp.eye(params.a_matrix.shape[0])
            + params.a_matrix @ params.a_matrix.T
        )

    @g.block(inputs=(), outputs="input_matrix")
    def input_matrix(params):
        return params.input_matrix

    # --- port-Hamiltonian dynamics -------------------------------------------
    @g.block(
        inputs=(
            "gradient_hamiltonian",
            "structure_matrix",
            "dissipation_matrix",
            "input_matrix",
            "u",
        ),
        outputs="x_dot",
    )
    def state_equation(
        params,
        gradient_hamiltonian,
        structure_matrix,
        dissipation_matrix,
        input_matrix,
        u,
    ):
        return (
            structure_matrix - dissipation_matrix
        ) @ gradient_hamiltonian + input_matrix @ u

    # --- normalization in ----------------------------------------------------
    @g.block(inputs="physical_ts", outputs="ts")
    def normalize_t(params, ts):
        return params.ts_normalizer.forward(ts)

    @g.block(inputs="physical_y0", outputs="y0")
    def normalize_y(params, y0):
        return params.ys_normalizer.forward(y0)

    @g.block(inputs="physical_us", outputs="us")
    def normalize_u(params, us):
        return params.us_normalizer.forward(us)

    @g.block(inputs="y0", outputs="x0")
    def make_x0(params, y0):
        return jnp.concatenate([y0, jnp.zeros(state_size - output_size)])

    # --- ODE solve -----------------------------------------------------------
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-5)

    @g.block(
        inputs=("ts", "x0", "us"), outputs="solution", uses="x, u -> x_dot"
    )
    def solve_ode(params, rhs, ts, x0, us):
        def vector_field(t, x, args):
            rhs, u_interp = args
            return rhs(x, u_interp.evaluate(t))

        coeffs = diffrax.backward_hermite_coefficients(ts, us)
        u_interp = diffrax.CubicInterpolation(ts, coeffs)
        return diffrax.diffeqsolve(
            terms=diffrax.ODETerm(vector_field),
            solver=diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=x0,
            args=(rhs, u_interp),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=stepsize_controller,
        )

    # --- output and normalization out ---------------------------------------
    @g.block(inputs="solution", outputs="xs")
    def extract_xs(params, solution):
        return solution.ys

    @g.block(inputs="xs", outputs="ys")
    def output_equation(params, xs):
        return xs[..., :output_size]

    @g.block(inputs="ys", outputs="physical_ys")
    def denormalize_ys(params, ys):
        return params.ys_normalizer.inverse(ys)

    graph = g.build()
    model = klax.finalize(GraphModel(params, graph))

    model = GraphModel(params, graph)

    model = klax.finalize(model)

    have: dict[str, Array] = {
        "physical_y0": jnp.array([1.0, 2.0, 3.0]),
        "physical_us": jnp.zeros((100, input_size)),
        "physical_ts": jnp.linspace(0, 1, 100),
    }
    want = ("solution", "physical_ys")

    @eqx.filter_jit
    def run(model, have, want):
        return model.run(have, want)

    jax.config.update("jax_log_compiles", True)
    out = run(model, have, want)
    out = run(model, have, want)

    print(out)

    dot = graph_to_dot(graph, have=("x"), want=("hamiltonian",))
    print(dot)
