import jax.random as jrandom
import numpy as np
import pytest
from equinox import tree_equal

from klax.datahandler import dataloader, split_data

def test_dataloader(getkey):
    # Sequence with one element
    x = jrandom.uniform(getkey(), (10,))
    data = (x,)
    generator = dataloader(data, key=getkey())
    assert isinstance(next(generator), tuple)
    assert len(next(generator)) == 1

    # Nested PyTree
    x = jrandom.uniform(getkey(), (10,))
    data = [x, (x, {"a": x, "b": x})]
    generator = dataloader(data, key=getkey())
    assert isinstance(next(generator), list)
    assert len(next(generator)) == 2
    assert isinstance(next(generator)[1], tuple)
    assert len(next(generator)[1]) == 2
    assert isinstance(next(generator)[1][1], dict)
    assert len(next(generator)[1][1]) == 2

    # Default batch size
    x = jrandom.uniform(getkey(), (33,))
    data = (x,)
    generator = dataloader(data, key=getkey())
    assert next(generator)[0].shape[0] == 32

    # Batch mask
    x = jrandom.uniform(getkey(), (10,))
    data = (x, (x, x))
    batch_axis = (0, (None, 0))
    generator = dataloader(data, 2, batch_axis, key=getkey())
    assert next(generator)[0].shape[0] == 2
    assert next(generator)[1][0].shape[0] == 10
    assert next(generator)[1][1].shape[0] == 2

    # No batch dimensions
    x = jrandom.uniform(getkey(), (10,))
    data = (x,)
    batch_axis = None
    with pytest.raises(
        ValueError, match="At least one leaf must have a batch dimension."
    ):
        generator = dataloader(data, batch_axis=batch_axis, key=getkey())
        next(generator)

    # Different batch dimensions
    x = jrandom.uniform(getkey(), (10,))
    y = jrandom.uniform(getkey(), (5,))
    data = (x, y)
    with pytest.raises(
        ValueError, match="All batched arrays must have equal batch dimension."
    ):
        generator = dataloader(data, key=getkey())
        next(generator)

    # Smaller data than batch dimension
    x = jrandom.uniform(getkey(), (10,))
    generator = dataloader(x, batch_size=128, key=getkey())
    assert next(generator).shape == (10,)


def test_split_data(getkey):

    # Expected usage
    batch_size = 20
    data = (
        jrandom.uniform(getkey(), (batch_size, 2)),
        [
            jrandom.uniform(getkey(), (3, batch_size, 2)),
            100.,
            'test',
            None,
        ],
    )
    proportions = (0.5, 0.25, 0.25)
    batch_axis = (0, 1)

    subsets = split_data(data, proportions, batch_axis, key=getkey())

    for s, p in zip(subsets, proportions):
        assert s[0].shape == (round(p*batch_size), 2)
        assert s[1][0].shape == (3, round(p*batch_size), 2)
        assert tree_equal(s[1][1:], data[1][1:])

    # Edge cases
    data = np.arange(10)
    s, = split_data(data, (1.,), key=getkey())
    assert np.array_equal(data, np.sort(s))

