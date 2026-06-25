Add xarray support to batch_data and split_data

`klax.batch_data` and `klax.split_data` previously crashed on `xr.Dataset`
and `xr.DataArray` inputs because the broadcast in `broadcast_and_get_size`
tried to rebuild an xarray container with integer leaves, violating
xarray's dim/coord invariants.

Refactor the data handlers to treat xarray containers as opaque leaves and
dispatch slicing by leaf type. `batch_axes` leaves now accept `int | str |
None`: `int` is a positional axis on an array leaf, `str` is a dim name on
an xarray leaf, `None` skips a subtree. A `str` spec on a non-xarray leaf
and an `int` spec on an xarray leaf both raise a `TypeError` with the
available alternatives spelled out.

xarray is an optional dependency. `klax._datahandler` imports it behind a
`_HAS_XARRAY` flag; if it is missing, the new code paths are dead and
plain-pytree batching behaves exactly as before. A new
`xarray-experimental` extra wires up `xarray` and `gdm-xarray-jax` for
users who want named-dim batching (and `xarray_jax`-backed pytree
registration when JIT'ing a loss that takes xarray batches).

- klax/_datahandler.py: add `_is_xarray`, `_is_leaf`, `_normalize_one`,
  `_broadcast_spec`, `_leaf_size`, `_leaf_take`; rewrite
  `broadcast_and_get_size`, `batch_data`, `split_data` around them; widen
  the `Batcher` Protocol; update docstrings with xarray examples.
- klax/_training.py, klax/_logging.py: widen `batch_axes` type annotation
  to `PyTree[int | str | None]` and update the `fit` docstring.
- pyproject.toml: add `xarray-experimental` extra (xarray + gdm-xarray-jax
  pinned via uv source to the upstream tag).
- tests/test_datahandler.py: add `TestBatchDataXArray` and
  `TestSplitDataXArray` (Dataset, DataArray, mixed pytrees, None
  passthrough, error cases) gated on `pytest.importorskip("xarray")`.
- tests/test_training.py: add `test_default_behavior_with_xarray` end-to-end
  with `xarray_jax` registration.
