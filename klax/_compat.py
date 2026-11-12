"""Optional dependency imports."""

import functools
import importlib.util

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


@functools.cache
def get_matplotlib():
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(
            "This feature requires 'matplotlib'. "
            "Install with: pip install klax[plot]."
        ) from exc
    return matplotlib


HAS_TQDM = importlib.util.find_spec("tqdm") is not None


@functools.cache
def get_tqdm():
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError(
            "This feature requires 'tqdm'. "
            "Install with: pip install klax[tqdm]."
        ) from exc
    return tqdm
