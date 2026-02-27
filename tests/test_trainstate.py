from collections.abc import Generator

import jax
import jax.numpy as jnp
import optax
import pytest
from jaxtyping import PyTreeDef

from klax import make_view


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


class TestTrainingView:
    def test_make_view_wraps_optimizer_and_sets_defs(self):
        model = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}
        opt_state = {"m": {"w": jnp.zeros(2), "b": jnp.zeros(())}}
        optimizer = optax.adam(1e-3)

        view = make_view(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            batch=dummy_batch(),
            aux=0,
            loss=object(),
            steps=5,
        )

        # Leaves and tree defs are set
        m_leaves, m_treedef = jax.tree.flatten(model)
        o_leaves, o_treedef = jax.tree.flatten(opt_state)
        assert view._state.model_leaves == m_leaves
        assert view._state.opt_state_leaves == o_leaves
        assert view._static.model_tree_def == m_treedef
        assert view._static.opt_state_tree_def == o_treedef

        # Optimizer is wrapped with extra args support
        assert isinstance(
            view._static.optimizer, optax.GradientTransformationExtraArgs
        )

        # If already wrapped, keep identity
        wrapped = optax.with_extra_args_support(optax.sgd(1.0))
        view2 = make_view(
            model=model,
            optimizer=wrapped,
            opt_state=opt_state,
            batch=dummy_batch(),
            aux=0,
            loss=object(),
            steps=3,
        )
        assert view2._static.optimizer is wrapped

    def test_assemble_disassemble_model_and_opt_state(self):
        model = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}
        opt_state = {"m": {"w": jnp.zeros(2), "b": jnp.zeros(())}}
        optimizer = optax.adam(1e-3)

        view = make_view(
            model, optimizer, opt_state, dummy_batch(), 0, object(), 2
        )

        model_round = view._static.assemble_model(view._state.model_leaves)
        tree_allclose(model_round, model)

        leaves = view._static.disassemble_model(model_round)
        assert leaves == view._state.model_leaves

        opt_round = view._static.assemble_opt_state(
            view._state.opt_state_leaves
        )
        tree_allclose(opt_round, opt_state)

        o_leaves = view._static.disassemble_opt_state(opt_round)
        assert o_leaves == view._state.opt_state_leaves

    def test_disassemble_model_mismatch_raises(self):
        model = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}
        other = {"w": jnp.array([1.0, 2.0]), "c": jnp.array(1.0)}
        opt_state = {"m": {"w": jnp.zeros(2), "b": jnp.zeros(())}}
        optimizer = optax.adam(1e-3)

        view = make_view(
            model, optimizer, opt_state, dummy_batch(), 0, object(), 2
        )

        with pytest.raises(ValueError, match="Model structure changed"):
            view._static.disassemble_model(other)

    def test_disassemble_opt_state_mismatch_raises(self):
        model = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}
        opt_state = {"m": {"w": jnp.zeros(2), "b": jnp.zeros(())}}
        other_opt = {"m": {"w": jnp.zeros(2), "c": jnp.zeros(())}}
        optimizer = optax.adam(1e-3)

        view = make_view(
            model, optimizer, opt_state, dummy_batch(), 0, object(), 2
        )

        with pytest.raises(ValueError, match="Opt state structure changed"):
            view._static.disassemble_opt_state(other_opt)

    def test_trainingview_model_property_caching_and_setter(self):
        model = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}
        opt_state = {"m": {"w": jnp.zeros(2), "b": jnp.zeros(())}}
        optimizer = optax.adam(1e-3)

        view = make_view(
            model, optimizer, opt_state, dummy_batch(), 0, object(), 2
        )

        # Getter assembles and caches
        m1 = view.model
        tree_allclose(m1, model)

        # Setter updates leaves and cache
        new_model = {"w": model["w"] + 1, "b": model["b"] + 1}
        view.model = new_model
        tree_allclose(view.model, new_model)
        roundtrip = view._static.assemble_model(view._state.model_leaves)
        tree_allclose(roundtrip, new_model)

    def test_trainingview_opt_state_getter_setter(self):
        model = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}
        opt_state = {"m": {"w": jnp.zeros(2), "b": jnp.zeros(())}}
        optimizer = optax.adam(1e-3)

        view = make_view(
            model, optimizer, opt_state, dummy_batch(), 0, object(), 2
        )

        # Getter assembles and returns opt state
        os1 = view.opt_state
        tree_allclose(os1, opt_state)

        # Setter updates leaves and cache
        new_opt = {"m": {"w": jnp.ones(2), "b": jnp.ones(())}}
        view.opt_state = new_opt
        tree_allclose(view.opt_state, new_opt)
        roundtrip = view._static.assemble_opt_state(
            view._state.opt_state_leaves
        )
        tree_allclose(roundtrip, new_opt)

    def test_static_properties(self):
        model = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}
        opt_state = {"m": {"w": jnp.zeros(2), "b": jnp.zeros(())}}
        optimizer = optax.adam(1e-3)
        loss = object()
        aux = 0
        steps = 2
        view = make_view(
            model,
            optimizer,
            opt_state,
            dummy_batch(),
            aux=aux,
            loss=loss,
            steps=steps,
        )

        assert view.model_tree_def == jax.tree.structure(model)
        with pytest.raises(AttributeError):
            view.model_tree_def = "something"

        assert isinstance(
            view.optimizer, optax.GradientTransformationExtraArgs
        )
        with pytest.raises(AttributeError):
            view.optimizer = "something"

        assert isinstance(view.opt_state_tree_def, PyTreeDef)
        with pytest.raises(AttributeError):
            view.opt_state_tree_def = "something"

        assert isinstance(view.batch, Generator)
        with pytest.raises(AttributeError):
            view.batch = "something"

        assert view.aux == aux
        with pytest.raises(AttributeError):
            view.aux_tree_def = "something"

        assert view.loss is loss
        with pytest.raises(AttributeError):
            view.loss = "something"

        assert view.steps == steps
        with pytest.raises(AttributeError):
            view.steps = "something"
