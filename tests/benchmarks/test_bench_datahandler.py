import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import klax
from klax._compat import HAS_XARRAY, get_xarray

# ===---------------------------------------------------------------------=== #
# klax.batch_data
# ===---------------------------------------------------------------------=== #


class TestBatchData:
    @pytest.mark.benchmark
    @staticmethod
    def test_convert_to_numpy(getkey, benchmark):
        x = jr.uniform(getkey(), (1000,))
        generator = klax.batch_data(x, batch_size=32, key=getkey())
        benchmark(next, generator)

    @pytest.mark.benchmark
    @staticmethod
    def test_jax_array(getkey, benchmark):
        x = jr.uniform(getkey(), (1000,))
        generator = klax.batch_data(
            x, batch_size=32, key=getkey(), convert_to_numpy=False
        )
        benchmark(next, generator)

    @pytest.mark.skipif(not HAS_XARRAY, reason="needs xarray and xarray_jax")
    @pytest.mark.benchmark
    @staticmethod
    def test_xarray_convert_to_numpy(getkey, benchmark):
        xr, _ = get_xarray()
        data = xr.DataArray(
            jr.uniform(getkey(), (1000,)),
            coords={"batch": jnp.arange(1000)},
            dims="batch",
        )

        # Test that the data is still the same, but the 'batch' axis was still
        # sorted. Hence, in direct comparsion the DataArrays are equal, but
        # they are still in different order.
        generator = klax.batch_data(
            data, batch_size=32, batch_axes="batch", key=getkey()
        )
        benchmark(next, generator)

    @pytest.mark.skipif(not HAS_XARRAY, reason="needs xarray and xarray_jax")
    @pytest.mark.benchmark
    @staticmethod
    def test_xarray_jax_array(getkey, benchmark):
        xr, _ = get_xarray()
        data = xr.DataArray(
            jr.uniform(getkey(), (1000,)),
            coords={"batch": jnp.arange(1000)},
            dims="batch",
        )

        # Test that the data is still the same, but the 'batch' axis was still
        # sorted. Hence, in direct comparsion the DataArrays are equal, but
        # they are still in different order.
        generator = klax.batch_data(
            data,
            batch_size=32,
            batch_axes="batch",
            key=getkey(),
            convert_to_numpy=False,
        )
        benchmark(next, generator)
