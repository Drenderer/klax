from jax.nn.initializers import uniform, he_normal
import jax
import jax.numpy as jnp
import jax.random as jrandom
import paramax as px
from klax.nn import Linear, InputSplitLinear, MLP
from klax.wrappers import NonNegative


def test_linear(getkey, getwrap):
    # Zero input shape
    linear = Linear(0, 4, uniform(), key=getkey())
    x = jrandom.normal(getkey(), (0,))
    assert linear(x).shape == (4,)

    # Zero output shape
    linear = Linear(4, 0, uniform(), key=getkey())
    x = jrandom.normal(getkey(), (4,))
    assert linear(x).shape == (0,)

    # Positional arguments
    linear = Linear(3, 4, uniform(), key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    # Some keyword arguments
    linear = Linear(3, out_features=4, weight_init=uniform(), key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    # All keyword arguments
    linear = Linear(in_features=3, out_features=4, weight_init=uniform(), key=getkey())
    x = jrandom.normal(getkey(), (3,))
    assert linear(x).shape == (4,)

    # Scalar shapes
    linear = Linear("scalar", 2, uniform(), key=getkey())
    x = jrandom.normal(getkey(), ())
    assert linear(x).shape == (2,)

    linear = Linear(2, "scalar", uniform(), key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert linear(x).shape == ()

    # Wrappers
    linear = Linear(
        3, 4, uniform(), weight_wrap=getwrap, bias_wrap=getwrap, key=getkey()
    )
    x = jrandom.normal(getkey(), (3,))
    assert jnp.all(px.unwrap(linear)(x) == 0.0)

    # Data type
    linear = Linear(2, "scalar", uniform(), key=getkey(), dtype=jnp.float16)
    x = jrandom.normal(getkey(), (2,), dtype=jnp.float16)
    assert linear(x).dtype == jnp.float16

    linear = Linear(
        2,
        "scalar",
        he_normal(),  # since uniform does not accept complex numbers
        key=getkey(),
        dtype=jnp.complex64,
    )
    x = jrandom.normal(getkey(), (2,), dtype=jnp.complex64)
    assert linear(x).dtype == jnp.complex64


def test_input_split_linear(getkey):
    input_split_linear = InputSplitLinear([3, 2], 4, uniform(), key=getkey())
    y = jrandom.normal(getkey(), (3,))
    z = jrandom.normal(getkey(), (2,))
    assert input_split_linear(y, z).shape == (4,)

    # Scalar shapes
    input_split_linear = InputSplitLinear(
        ("scalar", 2), 3, (uniform(), uniform()), key=getkey()
    )
    y = jrandom.normal(getkey(), ())
    z = jrandom.normal(getkey(), (2,))
    assert input_split_linear(y, z).shape == (3,)

    input_split_linear = InputSplitLinear(
        [2, 3], "scalar", uniform(), key=getkey()
    )
    y = jrandom.normal(getkey(), (2,))
    z = jrandom.normal(getkey(), (3,))
    assert input_split_linear(y, z).shape == ()

    # Weight wrappers
    input_split_linear = InputSplitLinear(
        [2, 3],
        "scalar",
        uniform(),
        weight_wraps=[NonNegative, None],
        key=getkey(),
    )
    assert isinstance(input_split_linear.weights[0], NonNegative)
    assert isinstance(input_split_linear.weights[1], jax.Array)

    # Data types
    input_split_linear = InputSplitLinear(
        (2, 3), "scalar", uniform(), key=getkey(), dtype=jnp.float16
    )
    y = jrandom.normal(getkey(), (2,), dtype=jnp.float16)
    z = jrandom.normal(getkey(), (3,), dtype=jnp.float16)
    assert input_split_linear(y, z).dtype == jnp.float16


def test_mlp(getkey):
    mlp = MLP(2, 3, 2 * [8], uniform(), key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)

    mlp = MLP(
        in_size=2,
        out_size=3,
        width_sizes=2 * [8],
        weight_init=uniform(),
        bias_init=uniform(),
        key=getkey(),
    )
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)

    mlp = MLP("scalar", 2, 2 * [2], uniform(), key=getkey())
    x = jrandom.normal(getkey(), ())
    assert mlp(x).shape == (2,)

    mlp = MLP(2, "scalar", 2 * [2], uniform(), key=getkey())
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == ()
    assert [mlp.layers[i].use_bias for i in range(0, 3)] == [True, True, True]

    mlp = MLP(
        2, 3, 2 * [8], uniform(), use_bias=False, use_final_bias=True, key=getkey()
    )
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)
    assert [mlp.layers[i].use_bias for i in range(0, 3)] == [False, False, True]

    mlp = MLP(
        2, 3, 2 * [8], uniform(), use_bias=True, use_final_bias=False, key=getkey()
    )
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)
    assert [mlp.layers[i].use_bias for i in range(0, 3)] == [True, True, False]

    mlp = MLP(
        2, 3, [4, 8], uniform(), use_bias=True, use_final_bias=False, key=getkey()
    )
    x = jrandom.normal(getkey(), (2,))
    assert mlp(x).shape == (3,)
    assert [mlp.layers[i].in_features for i in range(0, 3)] == [2, 4, 8]
    assert [mlp.layers[i].out_features for i in range(0, 3)] == [4, 8, 3]
