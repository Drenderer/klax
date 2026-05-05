# Copyright 2025 The Klax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import equinox as eqx
import jax
import jax.random as jrandom
import numpy as np
import pytest

import klax


class TestBatchData:
    def test_with_single_array(self, getkey):
        x = jrandom.uniform(getkey(), (64,))
        data = x
        generator = klax.batch_data(data, batch_size=32, key=getkey())
        assert jax.tree.structure(next(generator)) == jax.tree.structure(data)

    def test_with_nested_pytree(self, getkey):
        x = jrandom.uniform(getkey(), (10,))
        data = [x, (x, {"a": x, "b": x})]
        generator = klax.batch_data(data, batch_size=32, key=getkey())
        assert jax.tree.structure(next(generator)) == jax.tree.structure(data)

    def test_batch_size(self, getkey):
        x = jrandom.uniform(getkey(), (33,))
        data = x
        generator = klax.batch_data(data, batch_size=32, key=getkey())
        assert next(generator).shape[0] == 32

    def test_batch_size_larger_than_data(self, getkey):
        x = jrandom.uniform(getkey(), (10,))
        data = x
        generator = klax.batch_data(data, batch_size=128, key=getkey())
        assert next(generator).shape == (10,)

    def test_different_batch_axes(self, getkey):
        x = jrandom.uniform(getkey(), (10,))
        data = (x, (x, x))
        batch_axes = (0, (None, 0))
        generator = klax.batch_data(
            data, batch_size=2, batch_axes=batch_axes, key=getkey()
        )
        assert next(generator)[0].shape[0] == 2
        assert next(generator)[1][0].shape[0] == 10
        assert next(generator)[1][1].shape[0] == 2

    def test_no_batch_axes(self, getkey):
        x = jrandom.uniform(getkey(), (10,))
        data = (x,)
        batch_axes = None
        generator = klax.batch_data(
            data, batch_size=2, batch_axes=batch_axes, key=getkey()
        )
        assert next(generator) == data

    def test_different_batch_sizes(self, getkey):
        x = jrandom.uniform(getkey(), (10,))
        y = jrandom.uniform(getkey(), (5,))
        data = (x, y)
        generator = klax.batch_data(data, batch_size=32, key=getkey())
        with pytest.raises(
            ValueError, match="All batched arrays must have equal batch sizes."
        ):
            next(generator)


class TestSplitData:
    def test_split_data(self, getkey):
        batch_size = 20
        data = (
            jrandom.uniform(getkey(), (batch_size, 2)),
            [
                jrandom.uniform(getkey(), (3, batch_size, 2)),
                100.0,
                "test",
                None,
            ],
        )
        proportions = (2, 1, 1)
        batch_axes = (0, 1)
        subsets = klax.split_data(data, proportions, batch_axes, key=getkey())

        for s, p in zip(subsets, (0.5, 0.25, 0.25)):
            assert s[0].shape == (round(p * batch_size), 2)
            assert s[1][0].shape == (3, round(p * batch_size), 2)
            assert eqx.tree_equal(s[1][1:], data[1][1:])

    def test_with_singleton_split(self, getkey):
        data = np.arange(10)
        (s,) = klax.split_data(data, (1,), key=getkey())
        assert np.array_equal(data, np.sort(s))

    def test_with_zero_proportion(self, getkey):
        data = np.arange(10)
        with pytest.raises(ValueError):
            klax.split_data(data, (-1.0,), key=getkey())
