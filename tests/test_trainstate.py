import jax
import jax.numpy as jnp
import optax
import pytest

from klax._trainstate import TrainingView, make_state_and_static


def tree_allclose(a, b):
    la, ta = jax.tree.flatten(a)
    lb, tb = jax.tree.flatten(b)
    assert ta == tb
    for xa, xb in zip(la, lb):
        if isinstance(xa, jnp.ndarray) or hasattr(xa, "shape"):
            assert jnp.allclose(xa, xb)
        else:
            assert xa == xb


def dummy_batch():
    while True:
        yield {"x": jnp.array([0.0])}


class TestTrainstate:
    def test_make_state_and_static_wraps_optimizer_and_sets_defs(self):
        model = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}
        opt_state = {"m": {"w": jnp.zeros(2), "b": jnp.zeros(())}}
        optimizer = optax.adam(1e-3)

        state, static = make_state_and_static(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            batch=dummy_batch(),
            batch_axes=0,
            loss=object(),
            steps=5,
        )

        # Leaves and tree defs are set
        m_leaves, m_treedef = jax.tree.flatten(model)
        o_leaves, o_treedef = jax.tree.flatten(opt_state)
        assert state.model_leaves == m_leaves
        assert state.opt_state_leaves == o_leaves
        assert static.model_tree_def == m_treedef
        assert static.opt_state_tree_def == o_treedef

        # Optimizer is wrapped with extra args support
        assert isinstance(
            static.optimizer, optax.GradientTransformationExtraArgs
        )

        # If already wrapped, keep identity
        wrapped = optax.with_extra_args_support(optax.sgd(1.0))
        _, static2 = make_state_and_static(
            model=model,
            optimizer=wrapped,
            opt_state=opt_state,
            batch=dummy_batch(),
            batch_axes=0,
            loss=object(),
            steps=3,
        )
        assert static2.optimizer is wrapped

    def test_assemble_disassemble_model_and_opt_state(self):
        model = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}
        opt_state = {"m": {"w": jnp.zeros(2), "b": jnp.zeros(())}}
        optimizer = optax.adam(1e-3)

        state, static = make_state_and_static(
            model, optimizer, opt_state, dummy_batch(), 0, object(), 2
        )

        model_round = static.assemble_model(state.model_leaves)
        tree_allclose(model_round, model)

        leaves = static.disassemble_model(model_round)
        assert leaves == state.model_leaves

        opt_round = static.assemble_opt_state(state.opt_state_leaves)
        tree_allclose(opt_round, opt_state)

        o_leaves = static.disassemble_opt_state(opt_round)
        assert o_leaves == state.opt_state_leaves

    def test_disassemble_model_mismatch_raises(self):
        model = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}
        other = {"w": jnp.array([1.0, 2.0]), "c": jnp.array(1.0)}
        opt_state = {"m": {"w": jnp.zeros(2), "b": jnp.zeros(())}}
        optimizer = optax.adam(1e-3)

        _, static = make_state_and_static(
            model, optimizer, opt_state, dummy_batch(), 0, object(), 2
        )

        with pytest.raises(ValueError, match="Model structure changed"):
            static.disassemble_model(other)

    def test_disassemble_opt_state_mismatch_raises(self):
        model = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}
        opt_state = {"m": {"w": jnp.zeros(2), "b": jnp.zeros(())}}
        other_opt = {"m": {"w": jnp.zeros(2), "c": jnp.zeros(())}}
        optimizer = optax.adam(1e-3)

        _, static = make_state_and_static(
            model, optimizer, opt_state, dummy_batch(), 0, object(), 2
        )

        with pytest.raises(ValueError, match="Opt state structure changed"):
            static.disassemble_opt_state(other_opt)

    def test_trainingview_model_property_caching_and_setter(self):
        model = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}
        opt_state = {"m": {"w": jnp.zeros(2), "b": jnp.zeros(())}}
        optimizer = optax.adam(1e-3)

        state, static = make_state_and_static(
            model, optimizer, opt_state, dummy_batch(), 0, object(), 2
        )

        view = TrainingView(state, static)
        # Getter assembles and caches
        m1 = view.model
        tree_allclose(m1, model)

        # Setter updates leaves and cache
        new_model = {"w": model["w"] + 1, "b": model["b"] + 1}
        view.model = new_model
        tree_allclose(view.model, new_model)
        roundtrip = static.assemble_model(state.model_leaves)
        tree_allclose(roundtrip, new_model)

    def test_trainingview_opt_state_getter_setter(self):
        model = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}
        opt_state = {"m": {"w": jnp.zeros(2), "b": jnp.zeros(())}}
        optimizer = optax.adam(1e-3)

        state, static = make_state_and_static(
            model, optimizer, opt_state, dummy_batch(), 0, object(), 2
        )

        view = TrainingView(state, static)
        # Getter assembles and returns opt state
        os1 = view.opt_state
        tree_allclose(os1, opt_state)

        # Setter updates leaves and cache
        new_opt = {"m": {"w": jnp.ones(2), "b": jnp.ones(())}}
        view.opt_state = new_opt
        tree_allclose(view.opt_state, new_opt)
        roundtrip = static.assemble_opt_state(state.opt_state_leaves)
        tree_allclose(roundtrip, new_opt)
