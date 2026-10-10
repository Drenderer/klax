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


def bregman_divergence(fun, x, y):
    return fun(x) - fun(y) - jnp.dot(jax.grad(fun)(y), (x - y))


def add_quadratic(f_bar, _x, x_star):
    dx = _x - x_star
    return f_bar + 0.5 * jnp.inner(dx, dx)


def diffeqsolve(params, rhs, ts, x0, us):
    def vector_field(t, x, args):
        params, rhs, u_interp = args
        return rhs(params, x, u_interp.evaluate(t))

    coeffs = diffrax.backward_hermite_coefficients(ts, us)
    u_interp = diffrax.CubicInterpolation(ts, coeffs)
    return diffrax.diffeqsolve(
        terms=diffrax.ODETerm(vector_field),
        solver=diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=x0,
        args=(params, rhs, u_interp),
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-5),
    )


if __name__ == "__main__":
    input_size, state_size, output_size = 2, 6, 3
    key = jr.PRNGKey(0)
    params = make_params(key, input_size, state_size, output_size)

    g = GraphBuilder()

    # --- Hamiltonian ---------------------------------------------------------
    @g.block()
    def f(params, _x):
        return params.ficnn(_x)

    @g.block(uses="f(_x)")
    def f_bar(params, f, _x):
        return bregman_divergence(f, _x, params.x_star)

    @g.block()
    def hamiltonian(params, f_bar, _x):
        return add_quadratic(f_bar, _x, params.x_star)

    @g.block(uses="hamiltonian(_x)")
    def gradient_hamiltonian(params, hamiltonian, _x):
        return jax.grad(hamiltonian)(_x)

    # --- system matrices -----------------------------------------------------
    @g.block()
    def structure_matrix(params):
        return params.a_matrix - params.a_matrix.T

    @g.block()
    def dissipation_matrix(params):
        return (
            params.epsilon * jnp.eye(params.a_matrix.shape[0])
            + params.a_matrix @ params.a_matrix.T
        )

    @g.block()
    def input_matrix(params):
        return params.input_matrix

    # --- port-Hamiltonian dynamics -------------------------------------------
    @g.block()
    def _x_dot(
        params,
        gradient_hamiltonian,
        structure_matrix,
        dissipation_matrix,
        input_matrix,
        _u,
    ):
        return (
            structure_matrix - dissipation_matrix
        ) @ gradient_hamiltonian + input_matrix @ _u

    # --- normalization in ----------------------------------------------------
    @g.block(outputs="_ts")
    def normalize_t(params, ts):
        return params.ts_normalizer.forward(ts)

    @g.block(outputs="_y0")
    def normalize_y(params, y0):
        return params.ys_normalizer.forward(y0)

    @g.block(outputs="_us")
    def normalize_u(params, us):
        return params.us_normalizer.forward(us)

    @g.block(outputs="_x0")
    def make_x0(params, _y0):
        return jnp.concatenate([_y0, jnp.zeros(state_size - output_size)])

    # --- ODE solve -----------------------------------------------------------
    @g.block(uses="_x_dot(_x, _u)")
    def solution(params, rhs, _ts, _x0, _us):
        return diffeqsolve(params, rhs.unbound, _ts, _x0, _us)

    # --- output and normalization out ---------------------------------------
    @g.block(outputs="_xs")
    def extract_xs(params, solution):
        return solution.ys

    @g.block(outputs="_ys")
    def output_equation(params, _xs):
        return _xs[..., :output_size]

    @g.block(outputs="ys")
    def denormalize_ys(params, _ys):
        return params.ys_normalizer.inverse(_ys)

    graph = g.build()
    model = klax.finalize(GraphModel(params, graph))

    model = GraphModel(params, graph)

    model = klax.finalize(model)

    have: dict[str, Array] = {
        "y0": jnp.array([1.0, 2.0, 3.0]),
        "us": jnp.zeros((100, input_size)),
        "ts": jnp.linspace(0, 1, 100),
    }
    want = ("solution", "ys")

    @eqx.filter_jit
    def run(model, have, want):
        return model.run(have, want)

    jax.config.update("jax_log_compiles", True)
    out = run(model, have, want)
    out = run(model, have, want)

    print(out)

    dot = graph_to_dot(graph, have=("_x"), want=("hamiltonian",))
    print(dot)
