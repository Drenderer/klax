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
import jax.numpy as jnp
import numpy as np

import klax


def _get_pytress():
    fun = lambda x: x
    obj = object()

    tree = (
        jnp.array(1),
        jnp.array([1.0, 2.0]),
        fun,
        np.array(1),
        np.array([1.0, 2.0]),
        (True, 1, 1.0, 1 + 1j),
        obj,
    )
    like = (
        jnp.array(5),
        jnp.array([6.0, 7.0]),
        fun,
        np.array(5),
        np.array([6.0, 7.0]),
        (False, 6, 6.0, 6 + 6j),
        obj,
    )
    return tree, like


def test_text_serialize_filter_spec(tmp_path):
    tree, like = _get_pytress()
    file_path = tmp_path / "model.eqx"

    eqx.tree_serialise_leaves(
        file_path, tree, filter_spec=klax.text_serialize_filter_spec
    )

    tree_loaded = eqx.tree_deserialise_leaves(
        file_path, like, filter_spec=klax.text_deserialize_filter_spec
    )

    print(tree_loaded)
    print(tree)

    assert eqx.tree_equal(tree_loaded, tree)
