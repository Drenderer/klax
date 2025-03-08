import klax
import jax.numpy as jnp
import jax.random as jrandom
import pytest


def test_dataloader(getkey):
    # Sequence with one element
    x = jrandom.uniform(getkey(), (10,))
    data = (x,)
    generator = klax.dataloader(data, key=getkey()) 
    assert len(next(generator)) == 1

    # Nested sequence
    x = jrandom.uniform(getkey(), (10,))
    data = (x, (x, x))
    generator = klax.dataloader(data, key=getkey()) 
    assert len(next(generator)) == 2
    assert len(next(generator)[1]) == 2

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

    # No batch dimensions
    x = jrandom.uniform(getkey(), (10,))
    data = (x,)
    batch_mask = (False,)
    with pytest.raises(ValueError):
        generator = klax.dataloader(data, batch_mask=batch_mask, key=getkey()) 
        next(generator)

    # Different batch dimensions
    x = jrandom.uniform(getkey(), (10,))
    y = jrandom.uniform(getkey(), (5,))
    data = (x, y)
    with pytest.raises(ValueError):
        generator = klax.dataloader(data, key=getkey()) 
        next(generator)
