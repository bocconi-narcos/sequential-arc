"""
Colour‑selection strategies for ARC‑style 2‑D grids.

* Public API *
──────────────
Each selector fulfils the same minimal contract
    selector(grid: np.ndarray, **kwargs) -> int
which lets the action‑space bind parameters with ``functools.partial``.

All selectors:
    • accept a NumPy array of shape (H, W) with integer entries ∈ [0, num_colours‑1]
    • return one of those integer colour IDs
    • raise ``ValueError`` on obviously invalid inputs
"""

from __future__ import annotations

from functools import partial

import numpy as np
from functools import partial
from scipy.ndimage import label

from .select import GridSelector


class ColorSelector:
    """Stateless helper that groups several colour-selection heuristics."""

    __slots__ = ()
    num_colours = 10
    _big = np.iinfo(np.int32).max

    # 4-connected neighbourhood (no diagonals)
    _FOUR = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=int)

    @staticmethod
    def _check_grid(grid: np.ndarray) -> None:
        if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
            raise ValueError("grid must be a 2-D NumPy array")
        if not np.issubdtype(grid.dtype, np.integer):
            raise ValueError("grid dtype must be an integer type")

    # ───────────────────────────────────────────────────────────── #
    # Histogram-based selectors
    # ───────────────────────────────────────────────────────────── #
    @staticmethod
    def colour(grid: np.ndarray, colour: int, **__) -> np.ndarray:
        """Mask of all cells equal to *colour*."""
        ColorSelector._check_grid(grid)
        # convert to int if necessary
        if not isinstance(colour, (int, np.integer)):
            raise ValueError("colour must be a non-negative integer")
        return colour
    
    colour_0, colour_0.__name__ = partial(colour, colour=0), "colour_0"
    colour_1, colour_1.__name__ = partial(colour, colour=1), "colour_1"
    colour_2, colour_2.__name__ = partial(colour, colour=2), "colour_2"
    colour_3, colour_3.__name__ = partial(colour, colour=3), "colour_3"
    colour_4, colour_4.__name__ = partial(colour, colour=4), "colour_4"
    colour_5, colour_5.__name__ = partial(colour, colour=5), "colour_5"
    colour_6, colour_6.__name__ = partial(colour, colour=6), "colour_6"
    colour_7, colour_7.__name__ = partial(colour, colour=7), "colour_7"
    colour_8, colour_8.__name__ = partial(colour, colour=8), "colour_8"
    colour_9, colour_9.__name__ = partial(colour, colour=9), "colour_9"
    colour_10, colour_10.__name__ = partial(colour, colour=10), "colour_10"

    
