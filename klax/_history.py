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

"""Training history."""

import json
import pickle
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

from jax import numpy as jnp
from jaxtyping import Array
from matplotlib.legend_handler import HandlerTuple

from ._compat import get_plot


class StepsAndValues(NamedTuple):
    steps: list[int]
    values: list[Any]


def _value_to_json(value: Any) -> Any:
    if isinstance(value, Array):
        return {
            "__jax_array__": True,
            "data": value.tolist(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return value


def _value_from_json(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__jax_array__"):
        return jnp.array(value["data"], dtype=value["dtype"]).reshape(
            value["shape"]
        )
    return value


class History:
    content: dict[str, StepsAndValues]
    total_time: float | None
    total_steps: int | None

    def __init__(self):
        self.content = defaultdict(lambda: StepsAndValues(steps=[], values=[]))
        self.total_time = None
        self.total_steps = None

    def append(self, key: str, step: int, value: Any) -> None:
        self.content[key].steps.append(step)
        self.content[key].values.append(value)

    def __getitem__(self, key: str) -> StepsAndValues:
        if key not in self.content:
            raise KeyError(f"Metric '{key}' not found in history.")
        return self.content[key]

    def __repr__(self) -> str:
        return f"History containing: {list(self.content.keys())}"

    def to_dict(self) -> dict:
        return {
            "content": {
                k: {
                    "steps": v.steps,
                    "values": [_value_to_json(x) for x in v.values],
                }
                for k, v in self.content.items()
            },
            "total_time": self.total_time,
            "total_steps": self.total_steps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "History":
        obj = cls()
        for k, v in d["content"].items():
            obj.content[k] = StepsAndValues(
                steps=v["steps"],
                values=[_value_from_json(x) for x in v["values"]],
            )
        obj.total_time = d["total_time"]
        obj.total_steps = d["total_steps"]
        return obj

    def save(self, path: str | Path) -> None:
        """Persist the history to disk using json.

        Args:
            path: Destination filepath where the history will be stored.

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        with path.open("w") as file:
            json.dump(payload, file, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "History":
        """Restore a history saved with :meth:`save`.

        Args:
            path: Filepath to load the serialized history from.

        Returns:
            A populated History instance.

        """
        path = Path(path)
        with path.open() as file:
            payload = json.load(file)

        return cls.from_dict(payload)

    def plot(
        self,
        *keys: str,
        exclude_keys: Iterable[str] | None = None,
        ax: Any = None,
        **kwargs: Any,
    ) -> None:
        """Plot stored metrics using matplotlib.

        Note:
            This method requires matplotlib.

        Args:
            keys: Metric names to plot. If empty, all metrics are plotted.
            exclude_keys: Metric names to exclude when plotting.
            ax: Matplotlib axes to plot into. If ``None`` then a new axis is
                created. (Defaults to None.)
            kwargs: Dictionary of keyword arguments passed to
                matplotlib's ``plot``.

        Raises:
            ImportError: If matplotlib is not installed.

        """
        plt = get_plot()

        exclude_keys = [] if exclude_keys is None else exclude_keys

        if ax is None:
            _, ax = plt.subplots()
            ax.set(
                xlabel="Step",
                ylabel="Metric",
                yscale="log",
                title="Training History",
            )
            ax.grid(True)
        keys = keys if keys else tuple(self.content.keys())
        artists = {}
        for name in keys:
            if name not in self.content:
                raise KeyError(
                    f"Key {name} not in History. Available keys: {list(self.content.keys())}"
                )
            if name in exclude_keys:
                continue
            steps, values = self.content[name]
            values = jnp.stack(values, axis=0)
            if values.ndim > 2:
                values = values.reshape(values.shape[0], -1)
            artist = ax.plot(steps, values, **kwargs)
            artists[name] = tuple(artist)
        ax.legend(
            artists.values(),
            artists.keys(),
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        )
        return ax
