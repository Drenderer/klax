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

"""Serialization deseriablizastion specs for `equinox.Module`."""

from typing import Any, BinaryIO

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def text_serialize_filter_spec(f: BinaryIO, x: Any) -> None:
    """Filter speficivation for serializing a leaf to text.

    Args:
        f: File-like object to write to.
        x: The leaf to save in the file.

    """
    if isinstance(x, (bool, complex, float, int, np.generic)):
        f.write(f"{str(x)}\n".encode())
    elif isinstance(x, jax.Array | np.ndarray):
        arr = x.flatten()
        f.write(f"{' '.join(str(float(v)) for v in arr)}\n".encode())
    else:
        pass


def text_deserialize_filter_spec(f: BinaryIO, x: Any) -> Any:
    """Filter speficication for deserializing a leaf from text.

    Args:
        f: File-like object to read from.
        x: The leaf for which to load data from the file.

    """
    line = f.readline().decode().strip()
    if isinstance(x, (bool, complex, float, int, np.generic)):
        return type(x)(line)
    elif isinstance(x, (jax.Array, jax.ShapeDtypeStruct)):
        return jnp.array(
            [float(v) for v in line.split()], dtype=x.dtype
        ).reshape(x.shape)
    elif isinstance(x, np.ndarray):
        return np.array(
            [float(v) for v in line.split()], dtype=x.dtype
        ).reshape(x.shape)
    else:
        return x
