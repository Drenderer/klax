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


xr = pytest.importorskip("xarray")


class TestBatchDataXArray:
    def _dataset(self, key, n_batch=100, n_features=4):
        import jax.numpy as jnp

        return xr.Dataset(
            {
                "y": (
                    ("batch", "i"),
                    jrandom.uniform(key, (n_batch, n_features)),
                ),
                "label": (("batch",), ["text"] * n_batch),
            },
            coords={"batch": jnp.arange(n_batch), "i": jnp.arange(n_features)},
        )

    def test_dataset_with_str_axis(self, getkey):
        data = self._dataset(getkey())
        generator = klax.batch_data(
            data, batch_size=10, batch_axes="batch", key=getkey()
        )
        batch = next(generator)
        assert isinstance(batch, xr.Dataset)
        assert batch.sizes["batch"] == 10
        assert batch.sizes["i"] == 4
        assert "batch" in batch.coords and "i" in batch.coords
        assert batch.coords["i"].size == 4

    def test_dataarray_with_str_axis(self, getkey):
        ds = self._dataset(getkey())
        data = ds["y"]
        generator = klax.batch_data(
            data, batch_size=8, batch_axes="batch", key=getkey()
        )
        batch = next(generator)
        assert isinstance(batch, xr.DataArray)
        assert batch.sizes["batch"] == 8
        assert batch.sizes["i"] == 4

    def test_mixed_pytree(self, getkey):
        ds = self._dataset(getkey())
        arr = jrandom.uniform(getkey(), (100, 3))
        data = {"x": arr, "y": ds}
        batch_axes = {"x": 0, "y": "batch"}
        generator = klax.batch_data(
            data, batch_size=10, batch_axes=batch_axes, key=getkey()
        )
        batch = next(generator)
        assert batch["x"].shape[0] == 10
        assert batch["y"].sizes["batch"] == 10

    def test_int_spec_on_xarray_raises(self, getkey):
        data = self._dataset(getkey())
        gen = klax.batch_data(data, batch_size=10, key=getkey())
        with pytest.raises(TypeError, match="must be a `str` dim name"):
            next(gen)

    def test_str_spec_on_array_raises(self, getkey):
        data = jrandom.uniform(getkey(), (10, 4))
        gen = klax.batch_data(
            data, batch_size=4, batch_axes="batch", key=getkey()
        )
        with pytest.raises(TypeError, match="only valid for xarray leaves"):
            next(gen)

    def test_unknown_dim_raises(self, getkey):
        data = self._dataset(getkey())
        gen = klax.batch_data(
            data, batch_size=10, batch_axes="missing", key=getkey()
        )
        with pytest.raises(ValueError, match="not present on xarray leaf"):
            next(gen)

    def test_none_passthrough_for_xarray(self, getkey):
        ds = self._dataset(getkey())
        arr = jrandom.uniform(getkey(), (100, 3))
        data = {"x": arr, "y": ds}
        batch_axes = {"x": 0, "y": None}
        generator = klax.batch_data(
            data, batch_size=10, batch_axes=batch_axes, key=getkey()
        )
        batch = next(generator)
        assert batch["x"].shape[0] == 10
        # y was not batched: same sizes as the original dataset
        assert batch["y"].sizes["batch"] == 100

    def test_convert_to_numpy_skips_xarray(self, getkey):
        ds = self._dataset(getkey())
        generator = klax.batch_data(
            ds,
            batch_size=10,
            batch_axes="batch",
            convert_to_numpy=True,
            key=getkey(),
        )
        batch = next(generator)
        assert isinstance(batch, xr.Dataset)


class TestSplitDataXArray:
    def _dataset(self, key, n_batch=20, n_features=2):
        import jax.numpy as jnp

        return xr.Dataset(
            {
                "y": (
                    ("batch", "i"),
                    jrandom.uniform(key, (n_batch, n_features)),
                ),
                "label": (("batch",), ["text"] * n_batch),
            },
            coords={"batch": jnp.arange(n_batch), "i": jnp.arange(n_features)},
        )

    def test_split_dataset(self, getkey):
        n_batch = 20
        data = self._dataset(getkey(), n_batch=n_batch)
        s1, s2 = klax.split_data(
            data, (3, 1), batch_axes="batch", key=getkey()
        )
        assert isinstance(s1, xr.Dataset) and isinstance(s2, xr.Dataset)
        assert s1.sizes["batch"] + s2.sizes["batch"] == n_batch
        assert s1.sizes["batch"] == 15 and s2.sizes["batch"] == 5
        assert s1.sizes["i"] == 2 and s2.sizes["i"] == 2

    def test_split_mixed(self, getkey):
        n_batch = 20
        ds = self._dataset(getkey(), n_batch=n_batch)
        arr = jrandom.uniform(getkey(), (n_batch, 5))
        data = {"x": arr, "y": ds}
        batch_axes = {"x": 0, "y": "batch"}
        s1, s2 = klax.split_data(
            data, (1, 1), batch_axes=batch_axes, key=getkey()
        )
        assert s1["x"].shape[0] == n_batch // 2
        assert s2["x"].shape[0] == n_batch // 2
        assert s1["y"].sizes["batch"] == n_batch // 2
        assert s2["y"].sizes["batch"] == n_batch // 2
