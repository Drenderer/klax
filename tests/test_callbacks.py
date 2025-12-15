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

import time

import jax.numpy as jnp
import pytest

import klax


def test_history_callback(
    dummy_train_state, dummy_train_static, dummy_metric_fn
):
    history = klax.HistoryCallback(
        metric_defs={"param_sum": dummy_metric_fn}, log_every=2
    )

    # On training start update
    history.on_training_start(dummy_train_state, dummy_train_static)
    assert history.last_start_time is not None
    assert list(history.metrics.keys()) == ["batch_loss", "param_sum"]
    assert len(history.metrics["batch_loss"]) == 1
    assert len(history.metrics["param_sum"]) == 1
    time.sleep(1e-6)  # Sleep to ensure that training time is greater than 0

    # First update
    history(dummy_train_state, dummy_train_static, 1, jnp.array(0.2))
    assert len(history.metrics["batch_loss"]) == 1
    assert len(history.metrics["param_sum"]) == 1

    # Second update
    history(dummy_train_state, dummy_train_static, 2, jnp.array(0.1))
    assert len(history.metrics["batch_loss"]) == 2
    assert len(history.metrics["param_sum"]) == 2

    # On training end update
    history.on_training_end(dummy_train_state, dummy_train_static, 2)

    assert history.training_time > 0.0


def test_history_callback_save_load(
    dummy_train_state, dummy_train_static, dummy_metric_fn, tmp_path
):
    history = klax.HistoryCallback(
        metric_defs={"param_sum": dummy_metric_fn}, log_every=1
    )

    history.on_training_start(dummy_train_state, dummy_train_static)
    history(dummy_train_state, dummy_train_static, 1, jnp.array(0.2))
    history(dummy_train_state, dummy_train_static, 2, jnp.array(0.1))
    history.on_training_end(dummy_train_state, dummy_train_static, 2)

    # Test save and load
    filepath = tmp_path / "some_dir/test_history.pkl"
    with pytest.raises(FileNotFoundError):
        history.save(filepath, create_dir=False)
    history.save(filepath, create_dir=True)
    with pytest.raises(FileExistsError):
        history.save(filepath, overwrite=False)
    history.save(filepath, overwrite=True)

    history2 = klax.HistoryCallback.load(filepath)

    # This is not a complete equality test!
    assert len(history2.metrics) == len(history.metrics)
