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

    @staticmethod
    def most_common(grid: np.ndarray) -> int:
        """Colour that occurs most frequently."""
        ColorSelector._check_grid(grid)
        counts = np.bincount(grid.ravel(),
                                minlength=ColorSelector.num_colours)
        return int(counts.argmax())

    @staticmethod
    def least_common(grid: np.ndarray) -> int:
        """Rarest colour that is **present** (ties broken arbitrarily)."""
        ColorSelector._check_grid(grid)
        counts = np.bincount(grid.ravel(),
                                minlength=ColorSelector.num_colours)
        counts[counts == 0] = ColorSelector._big
        return int(counts.argmin())

    @staticmethod
    def nth_most_common(grid: np.ndarray, *, rank: int) -> int:
        if rank < 0:
            raise ValueError("rank must be non-negative")
        ColorSelector._check_grid(grid)

        counts = np.bincount(grid.ravel(),
                                minlength=ColorSelector.num_colours)
        order = np.argsort(counts)[::-1]
        return int(order[rank]) if rank < len(order) else ColorSelector.least_common(grid)

    # rename the partials after defining nth_most_common
    second_most_common, second_most_common.__name__  = partial(nth_most_common, rank=1), "second_most_common"
    third_most_common, third_most_common.__name__    = partial(nth_most_common, rank=2), "third_most_common"
    fourth_most_common, fourth_most_common.__name__  = partial(nth_most_common, rank=3), "fourth_most_common"
    fifth_most_common, fifth_most_common.__name__    = partial(nth_most_common, rank=4), "fifth_most_common"
    sixth_most_common, sixth_most_common.__name__    = partial(nth_most_common, rank=5), "sixth_most_common"
    seventh_most_common, seventh_most_common.__name__ = partial(nth_most_common, rank=6), "seventh_most_common"
    eighth_most_common, eighth_most_common.__name__   = partial(nth_most_common, rank=7), "eighth_most_common"

    @staticmethod
    def nth_most_independent(grid: np.ndarray,
                                *,
                                rank: int,
                                connectivity: int = 4) -> int:
        """
        Return the colour whose single‐cell, non‐background components
        (under the given 4- or 8-connectivity) ranks `rank` in size.
        Rank 0 → most independent squares; rank=1 → second most; etc.
        If rank is out of range, returns the colour with the fewest.
        """
        if rank < 0:
            raise ValueError("rank must be non-negative")
        ColorSelector._check_grid(grid)

        # 1) count independent squares for each colour
        counts = np.zeros(ColorSelector.num_colours, dtype=int)
        for col in range(ColorSelector.num_colours):
            mask = GridSelector.independent_cells(grid, col, connectivity)
            counts[col] = int(mask.sum())

        # 2) sort colours by descending count (tie‐break by lower colour id)
        order = np.argsort(counts)[::-1]

        # 3) pick the rank-th, or the last one if rank >= num_colours
        idx = rank if rank < order.size else -1
        return int(order[idx])

    most_independent_cells, most_independent_cells.__name__ = partial(nth_most_independent, rank=0), "most_independent_cells"
    second_most_independent_cells, second_most_independent_cells.__name__ = partial(nth_most_independent, rank=1), "second_most_independent_cells"
    third_most_independent_cells, third_most_independent_cells.__name__ = partial(nth_most_independent, rank=2), "third_most_independent_cells"

    # ───────────────────────────────────────────────────────────── #
    # Shape-aware selector
    # ───────────────────────────────────────────────────────────── #
    @staticmethod
    def colour_of_nth_largest_shape(grid: np.ndarray, *, rank: int) -> int:
        if rank < 0:
            raise ValueError("rank must be non-negative")
        ColorSelector._check_grid(grid)

        sizes, colours_of_shapes = [], []
        for colour in np.unique(grid):
            mask = grid == colour
            lbl, num = label(mask, structure=ColorSelector._FOUR)
            if num == 0:
                continue
            component_sizes = np.bincount(lbl.ravel())[1:]
            sizes.extend(component_sizes)
            colours_of_shapes.extend([colour] * len(component_sizes))

        if not sizes:
            return ColorSelector.most_common(grid)

        order = np.argsort(sizes)[::-1]
        idx = order[rank] if rank < len(order) else order[-1]
        return int(colours_of_shapes[idx])

