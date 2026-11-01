"""Optional dependency imports."""

import functools
import importlib.util

HAS_PLOT = importlib.util.find_spec("matplotlib") is not None


@functools.cache
def get_plot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "This feature requires the 'plot' extra. "
            "Install with: pip install klax[plot]"
        ) from exc
    return plt


HAS_TQDM = importlib.util.find_spec("tqdm") is not None


@functools.cache
def get_tqdm():
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError(
            "This feature requires the 'tqdm' extra. "
            "Install with: pip install klax[tqdm]"
        ) from exc
    return tqdm
