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

"""Implements methods for handling data, such as batching and splitting."""

import typing
import warnings
from collections.abc import Generator, Sequence
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import PRNGKeyArray, PyTree


def _is_leaf(x: Any) -> bool:
    """Tree-traversal stop predicate used throughout this module.

    Stops at `None` (the "do not batch this subtree" sentinel).
    """
    return x is None


def _resolve_one(spec: Any, leaf: Any) -> int | None:
    """Resolve one `(spec, leaf)` pair to its normalized form.

    The normalized value is a single token that downstream code can carry in a
    pytree of the same shape as `data`:

    - `None`  — do not batch this leaf
    - `int`   — positional axis on an array leaf
    """
    if spec is None:
        return None
    if not isinstance(spec, (int, np.integer)):
        raise TypeError(
            f"batch_axes leaves must be `int` or `None`, "
            f"got {type(spec).__name__} ({spec!r})"
        )
    if not eqx.is_array(leaf):
        # Mirror the legacy `eqx.if_array` behavior: a non-array leaf paired
        # with an int spec is silently treated as "do not batch". This keeps
        # `batch_axes=0` ergonomic when users mix array data with scalars,
        # strings, or other non-array metadata in the same pytree.
        return None
    return int(spec)


def _broadcast_spec(spec: Any, subtree: Any) -> Any:
    """Broadcast a single spec across a (possibly nested) data subtree."""
    return jax.tree.map(
        lambda leaf: _resolve_one(spec, leaf),
        subtree,
        is_leaf=_is_leaf,
    )


def _leaf_size(leaf: Any, spec: int | None) -> int | None:
    if spec is None:
        return None
    return int(leaf.shape[spec])


def _leaf_take(leaf: Any, spec: int | None, indices: Any) -> Any:
    if spec is None:
        return leaf
    if spec == 0:
        return leaf[indices]
    if isinstance(leaf, np.ndarray):
        return np.take(leaf, np.asarray(indices), axis=spec)
    return jnp.take(leaf, indices, axis=spec)


def broadcast_and_get_size[T](
    data: PyTree[Any, "T"],
    batch_axes: PyTree[int | None, "T ..."],
) -> tuple[PyTree[int | None, "T"], int]:
    """Broadcast `batch_axes` to the same structure as `data` and get size.

    Args:
        data: PyTree of data. Leaves may be JAX/NumPy arrays or non-array
            Python values (which are passed through unchanged).
        batch_axes: PyTree of batch-axis specifications. Each leaf is one of:

            - `int`: positional axis on an array leaf.
            - `None`: the corresponding leaf or subtree in `data` is not
              batched.

            `batch_axes` must have the same structure as `data` or have
            `data` as a prefix. When `batch_axes` is a prefix, the spec at
            each prefix leaf is broadcast over the corresponding subtree of
            `data`: array leaves get the spec, non-array leaves (scalars,
            strings, ...) are silently treated as non-batched.

    Raises:
        TypeError: If a `batch_axes` leaf is neither an `int` nor `None`.
        ValueError: If not all batched arrays have equal batch sizes.

    Returns:
        Tuple of the resolved `batch_axes` (same structure as `data`,
        leaves are `int | None`) and the `dataset_size`.

    """
    resolved_axes = jax.tree.map(
        _broadcast_spec,
        batch_axes,
        data,
        is_leaf=_is_leaf,
    )

    sizes = jax.tree.map(
        _leaf_size,
        data,
        resolved_axes,
        is_leaf=_is_leaf,
    )
    leaves = [
        s for s in jax.tree.leaves(sizes, is_leaf=_is_leaf) if s is not None
    ]
    if not leaves:
        # No leaf in data has a batch dimension -> singelton data set
        dataset_size = 1
    else:
        if not all(v == leaves[0] for v in leaves):
            raise ValueError(
                f"All batched arrays must have equal batch sizes. {sizes=}"
            )
        dataset_size = leaves[0]

    return resolved_axes, dataset_size


@typing.runtime_checkable
class Batcher(Protocol):
    def __call__(
        self,
        data: PyTree[Any],
        batch_size: int,
        batch_axes: PyTree[int | None],
        *,
        key: PRNGKeyArray,
    ) -> Generator[PyTree[Any], None, None]:
        raise NotImplementedError


def batch_data[T](
    data: PyTree[Any, "T"],
    batch_size: int,
    batch_axes: PyTree[int | None] = 0,
    convert_to_numpy: bool = True,
    *,
    key: PRNGKeyArray,
) -> Generator[PyTree[Any, "T"], None, None]:
    """Create a `Generator` that draws subsets of data without replacement.

    The data can be any `PyTree` whose leaves are arrays or non-array Python
    values. `batch_axes` is a `PyTree` whose leaves are either positional
    axis indices (`int`) for array leaves or `None` for leaves that should
    not be batched. A generator is returned
    that indefinitely yields batches of data with size `batch_size`.
    Examples are drawn without replacement until the remaining dataset is
    smaller than `batch_size`, at which point the dataset is reshuffled and
    the process starts over.

    Example:
        ```python
        >>> import klax
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> x = jnp.array([1., 2.])
        >>> y = jnp.array([[1.], [2.]])
        >>> data = (x, {"a": 1.0, "b": y})
        >>> batch_axes = (0, {"a": None, "b": 0})
        >>> batch = klax.batch_data(
        ...     data,
        ...     32,
        ...     batch_axes,
        ...     key=jax.random.key(0)
        ... )
        ```

    Args:
        data: The data that shall be batched.
        batch_size: The number of examples in a batch.
        batch_axes: PyTree of batch-axis specifications. Each leaf is either
            an `int` (positional axis on an array leaf) or `None` (do not
            batch this subtree). `batch_axes` must have the same structure
            as `data` or have `data` as a prefix. (Defaults to `0`, meaning
            all array leaves in `data` are batched along their first
            dimension.)
        convert_to_numpy: If `True`, batched array leaves are converted to
            NumPy arrays before batching. Numpy's slicing is much faster
            than JAX's when running on CPU.
        key: A `jax.random.PRNGKey` used to provide randomness for batch
            generation. (Keyword only argument.)

    Returns:
        A `Generator` that yields a random batch of data.

    Yields:
        A `PyTree` with the same structure as `data`, where each batched
        leaf has been sliced to `batch_size` along its batch dimension.

    Note:
        Note that if the size of the dataset is smaller than `batch_size`,
        the used `batch_size` will be reduced.

    """
    resolved_axes, dataset_size = broadcast_and_get_size(data, batch_axes)

    # Convert to Numpy arrays. Numpy's slicing is much faster than JAX's, so
    # for fast model training steps this actually makes a huge difference!
    # However, be aware that this is likely only true if JAX runs on CPU.
    if convert_to_numpy:
        data = jax.tree.map(
            lambda x, spec: x if spec is None else np.array(x),
            data,
            resolved_axes,
            is_leaf=_is_leaf,
        )

    # Reduce batch size if the dataset has less examples than batch size
    batch_size = min(batch_size, dataset_size)

    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)  # Update key
        start, end = 0, batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield jax.tree.map(
                lambda leaf, spec: _leaf_take(leaf, spec, batch_perm),
                data,
                resolved_axes,
                is_leaf=_is_leaf,
            )
            start = end
            end = start + batch_size


def split_data[T: PyTree](
    data: T,
    proportions: Sequence[int | float],
    batch_axes: PyTree[int | None] = 0,
    *,
    key: PRNGKeyArray,
) -> tuple[T, ...]:
    """Split a `PyTree` of data into multiply randomly drawn subsets.

    This function is useful for splitting into training and test datasets.
    The axis of the split is controlled by the `batch_axes` argument, which
    specifies the batch axis for each leaf in `data` (positional `int` for
    array leaves, or `None` to leave a leaf unsplit).

    Example:
        ```python
        >>> import klax
        >>> import jax
        >>>
        >>> x = jax.numpy.array([1., 2., 3.])
        >>> data = (x, {"a": 1.0, "b": x})
        >>> s1, s2 = klax.split_data(
        ...     data,
        ...     (2, 1),
        ...     key=jax.random.key(0)
        ... )
        >>> s1
        (Array([1., 2.], dtype=float32), {'a': 1.0, 'b': Array([1., 2.], dtype=float32)})
        >>> s2
        (Array([3.], dtype=float32), {'a': 1.0, 'b': Array([3.], dtype=float32)})
        ```

    Args:
        data: Data that shall be split. It can be any `PyTree`.
        proportions: Proportions of the split that will be applied to the
            data, e.g., `(80, 20)` for a 80% to 20% split. The proportions
            must be non-negative.
        batch_axes: PyTree of batch-axis specifications. Each leaf is either
            an `int` (positional axis on an array leaf) or `None` (do not
            split this subtree). `batch_axes` must have the same structure
            as `data` or have `data` as a prefix. (Defaults to `0`.)
        key: A `jax.random.PRNGKey` used to provide randomness to the split.
            (Keyword only argument.)

    Returns:
        Tuple of `PyTrees`.

    """
    props = jnp.array(proportions, dtype=float)
    if props.ndim != 1:
        raise ValueError("Proportions must be a 1D Sequence.")
    if jnp.any(props < 0.0):
        raise ValueError("Proportions must be non-negative.")
    props = props / jnp.sum(props)

    resolved_axes, dataset_size = broadcast_and_get_size(data, batch_axes)

    indices = jnp.arange(dataset_size)
    perm = jr.permutation(key, indices)

    split_indices = jnp.round(
        jnp.cumsum(jnp.array(props[:-1]) * dataset_size)
    ).astype(int)
    sections = jnp.split(perm, split_indices)

    if not all(s.size for s in sections):
        warnings.warn("Proportions result in one or more empty subsets.")

    def get_subset(section):
        return jax.tree.map(
            lambda leaf, spec: _leaf_take(leaf, spec, section),
            data,
            resolved_axes,
            is_leaf=_is_leaf,
        )

    return tuple(get_subset(section) for section in sections)
