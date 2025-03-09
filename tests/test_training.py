import klax
import jax.numpy as jnp
import jax.random as jrandom
import pytest


def test_dataloader(getkey):
    # Sequence with one element
    x = jrandom.uniform(getkey(), (10,))
    data = (x,)
    generator = klax.dataloader(data, key=getkey()) 
    assert isinstance(next(generator), tuple)
    assert len(next(generator)) == 1

    # Nested PyTree
    x = jrandom.uniform(getkey(), (10,))
    data = [x, (x, {"a": x, "b": x})]
    generator = klax.dataloader(data, key=getkey()) 
    assert isinstance(next(generator), list)
    assert len(next(generator)) == 2
    assert isinstance(next(generator)[1], tuple)
    assert len(next(generator)[1]) == 2
    assert isinstance(next(generator)[1][1], dict)
    assert len(next(generator)[1][1]) == 2

    # Default batch size
    x = jrandom.uniform(getkey(), (33,))
    data = (x,)
    generator = klax.dataloader(data, key=getkey()) 
    assert next(generator)[0].shape[0] == 32

    # Batch mask
    x = jrandom.uniform(getkey(), (10,))
    data = (x, (x, x))
    batch_mask = (True, (False, True))
    generator = klax.dataloader(data, 2, batch_mask, key=getkey()) 
    assert next(generator)[0].shape[0] == 2
    assert next(generator)[1][0].shape[0] == 10
    assert next(generator)[1][1].shape[0] == 2

    # Different PyTree structures
    x = jrandom.uniform(getkey(), (10,))
    data = (x, x)
    batch_mask = (True, )
    with pytest.raises(
        ValueError,
        match="Arguments data and batch_mask must have equal PyTree structures."
    ):
        generator = klax.dataloader(data, batch_mask=batch_mask, key=getkey())
        next(generator)

    # No batch dimensions
    x = jrandom.uniform(getkey(), (10,))
    data = (x,)
    batch_mask = (False,)
    with pytest.raises(
        ValueError,
        match="At least one element must have batch dimension."
    ):
        generator = klax.dataloader(data, batch_mask=batch_mask, key=getkey()) 
        next(generator)

    # Different batch dimensions
    x = jrandom.uniform(getkey(), (10,))
    y = jrandom.uniform(getkey(), (5,))
    data = (x, y)
    with pytest.raises(
        ValueError,
        match="All batched arrays must have equal batch dimension."
    ):
        generator = klax.dataloader(data, key=getkey()) 
        next(generator)
