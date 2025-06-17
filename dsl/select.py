"""
Grid-Selection Utilities for ARC-Style Tasks
============================================

This module provides a lightweight, NumPy-friendly API for selecting
and analyzing regions within 2D integer grids, following ARC conventions.

Design Principles
-----------------
- **One selector per method**: Each public method implements exactly one
  selection primitive, named clearly to reflect its behavior.
- **Unified signature**: All selectors accept:
  
    grid:   np.ndarray of shape (H, W) and integer dtype  
    colour: int (optional) — ignored if unused  
    **kwargs: selector-specific parameters  

  and return either a boolean mask of shape (H, W), or a stack of masks
  of shape (N, H, W) for multi-region selectors.

- **Vectorized internally**: Leverages NumPy and SciPy for performance,
  no external DSL dependencies.
- **Minimal validation**: Only basic checks on grid shape and dtype;
  extra arguments are silently ignored if not needed.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from scipy.ndimage import convolve, label
from skimage.segmentation import find_boundaries

from dsl.utils.background import find_background_colour


# ───────────────────────────────────────────────────────────────────── #
# Helper constants & validators
# ───────────────────────────────────────────────────────────────────── #
_FOUR_CONN  = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=int)

_EIGHT_CONN = np.ones((3, 3), dtype=int)

def _check_grid(grid: np.ndarray) -> None:
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("grid must be a 2‑D NumPy array")
    if not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("grid dtype must be an integer type")

def _check_colour(colour: int) -> None:
    if not (isinstance(colour, (int, np.integer)) and colour >= 0):
        raise ValueError("colour must be a non‑negative integer")


# ───────────────────────────────────────────────────────────────────── #
# The selector library
# ───────────────────────────────────────────────────────────────────── #
class GridSelector:
    """Collection of selection primitives."""

    __slots__ = ("_min_geom",)

    def __init__(self, *, min_geometry: int = 2) -> None:
        self._min_geom = int(min_geometry)

    # ─────────────────────────────────────────────── #
    # 0  Trivial selection
    # ─────────────────────────────────────────────── #
    @staticmethod
    def all_cells(grid: np.ndarray, *_, **__) -> np.ndarray:
        """Mask with every cell set to *True*."""
        _check_grid(grid)
        return np.ones_like(grid, dtype=bool)

    # ─────────────────────────────────────────────── #
    # 1  Colour‑based
    # ─────────────────────────────────────────────── #
    @staticmethod
    def colour(grid: np.ndarray, colour: int, **__) -> np.ndarray:
        """Mask of all cells equal to *colour*."""
        _check_grid(grid)
        _check_colour(colour)
        return grid == colour
    
    colour, colour.__name__ = partial(colour), "colour"

    # ─────────────────────────────────────────────── #
    # 2  Connected components
    # ─────────────────────────────────────────────── #
    @staticmethod
    def components4(grid: np.ndarray, colour: int, **__) -> np.ndarray:
        """Returns a 2D mask of all 4-connected components of the given color."""
        mask = GridSelector.colour(grid, colour)
        lbl, n = label(mask, structure=_FOUR_CONN)
        if n == 0:
            return np.zeros_like(grid, dtype=bool)
        return lbl > 0  # Convert to boolean mask

    @staticmethod
    def components8(grid: np.ndarray, colour: int, **__) -> np.ndarray:
        """Returns a 2D mask of all 8-connected components of the given color."""
        mask = GridSelector.colour(grid, colour)
        lbl, n = label(mask, structure=_EIGHT_CONN)  # Use module-level constant
        if n == 0:
            return np.zeros_like(grid, dtype=bool)
        return lbl > 0  # Convert to boolean mask
    
    @staticmethod
    def nth_largest_shape(grid: np.ndarray, colour: int, connectivity: int, rank: int) -> np.ndarray:
        """
        Return the boolean mask of the nth-largest connected component of `colour`,
        using 4- or 8-connectivity. Rank 0 = largest, 1 = second largest, etc.
        Ties by component size are broken by the component whose top-leftmost cell
        (minimal row, then minimal column) is smallest.
        """
        # pick the right component extractor
        if connectivity == 4:
            comps = GridSelector.components4(grid, colour)
        elif connectivity == 8:
            comps = GridSelector.components8(grid, colour)
        else:
            raise ValueError("connectivity must be 4 or 8")

        n_comps = comps.shape[0]
        if rank < 0 or rank >= n_comps:
            # no such component → empty mask
            return np.zeros(grid.shape, dtype=bool)

        # compute sizes
        flat = comps.reshape(n_comps, -1)
        sizes = flat.sum(axis=1)

        # compute top-left position of each component
        positions = []
        for i in range(n_comps):
            coords = np.argwhere(comps[i])
            row_min = coords[:,0].min()
            # among cells in that minimal row, take the leftmost
            col_min = coords[coords[:,0] == row_min][:,1].min()
            positions.append((row_min, col_min))

        # sort indices by size desc, then row_min asc, then col_min asc
        idxs = list(range(n_comps))
        idxs.sort(key=lambda i: (-sizes[i], positions[i][0], positions[i][1]))

        # select the rank-th
        chosen = idxs[rank]
        return comps[chosen]


    @staticmethod
    def independent_cells(grid: np.ndarray,
                            colour: int,
                            connectivity: int = 4) -> np.ndarray:
        """
        Returns a boolean mask of shape (H, W) marking exactly those cells
        of value `colour` which form single‐cell (4‐connected) components
        among _all_ non‐background cells.
        """
        # 1) all cells of the requested colour (under given connectivity)
        colour_mask = GridSelector.colour(grid, colour)

        # 2) build non‐background mask
        _bg = find_background_colour(grid)
        non_bg = (grid != _bg)

        # 3) 4‐connected components of the non‐background mask
        #    (pass it through components4 by treating True as “1”)
        non_bg_comps = GridSelector.components4(non_bg.astype(int), 1)

        # 4) pick out those layers whose sum == 1 (i.e. singletons)
        singletons = [comp for comp in non_bg_comps if comp.sum() == 1]
        if not singletons:
            # no isolated squares at all
            return np.zeros_like(grid, dtype=bool)

        # union all those singleton‐layers into one mask
        singleton_mask = np.any(np.stack(singletons), axis=0)

        # 5) only keep those singletons that are of the requested colour
        return colour_mask & singleton_mask
    
    independent_cells_4, independent_cells_4.__name__ = partial(independent_cells, connectivity=4), "independent_cells_4"




    # ─────────────────────────────────────────────── #
    # 3  Outer / inner borders of each component
    # ─────────────────────────────────────────────── #
    @staticmethod
    def outer_border4(grid: np.ndarray, colour: int, **__) -> np.ndarray:
        """Returns a 2D mask of the outer border of all 4-connected components."""
        mask = GridSelector.colour(grid, colour)
        lbl, n = label(mask, structure=_FOUR_CONN)  # Use module-level constant
        if n == 0:
            return np.zeros_like(grid, dtype=bool)
        # Get all components
        components = np.stack([(lbl == i) for i in range(1, n + 1)])
        # Find boundaries for each component
        boundaries = np.array([find_boundaries(c, mode="outer") for c in components])
        # Combine all boundaries
        return np.any(boundaries, axis=0)

    @staticmethod
    def inner_border4(grid: np.ndarray, colour: int, **__) -> np.ndarray:
        """Returns a 2D mask of the inner border of all 4-connected components."""
        mask = GridSelector.colour(grid, colour)
        lbl, n = label(mask, structure=_FOUR_CONN)  # Use module-level constant
        if n == 0:
            return np.zeros_like(grid, dtype=bool)
        # Get all components
        components = np.stack([(lbl == i) for i in range(1, n + 1)])
        # Find boundaries for each component
        boundaries = np.array([find_boundaries(c, mode="inner") for c in components])
        # Combine all boundaries
        return np.any(boundaries, axis=0)
    
    @staticmethod
    def grid_border(grid: np.ndarray, colour: int, **__) -> np.ndarray:
        comps = np.zeros_like(grid, dtype=bool)
        comps[0, :] = True
        comps[-1, :] = True
        comps[:, 0] = True
        comps[:, -1] = True
        return comps


    @staticmethod
    def outer_border8(grid: np.ndarray, colour: int, **__) -> np.ndarray:
        """Returns a 2D mask of the outer border of all 8-connected components."""
        mask = GridSelector.colour(grid, colour)
        lbl, n = label(mask, structure=_EIGHT_CONN)  # Use module-level constant
        if n == 0:
            return np.zeros_like(grid, dtype=bool)
        # Get all components
        components = np.stack([(lbl == i) for i in range(1, n + 1)])
        # Find boundaries for each component
        boundaries = np.array([find_boundaries(c, mode="outer") for c in components])
        # Combine all boundaries
        return np.any(boundaries, axis=0)

    @staticmethod
    def inner_border8(grid: np.ndarray, colour: int, **__) -> np.ndarray:
        """Returns a 2D mask of the inner border of all 8-connected components."""
        mask = GridSelector.colour(grid, colour)
        lbl, n = label(mask, structure=_EIGHT_CONN)  # Use module-level constant
        if n == 0:
            return np.zeros_like(grid, dtype=bool)
        # Get all components
        components = np.stack([(lbl == i) for i in range(1, n + 1)])
        # Find boundaries for each component
        boundaries = np.array([find_boundaries(c, mode="inner") for c in components])
        # Combine all boundaries
        return np.any(boundaries, axis=0)

    grid_border, grid_border.__name__ = partial(grid_border), "grid_border"
    

    # ─────────────────────────────────────────────── #
    # 4  Adjacency
    # ─────────────────────────────────────────────── #
    @staticmethod
    def adjacent4(grid: np.ndarray, colour: int, *, contacts: int = 1) -> np.ndarray:
        """
        Cells *not* of `colour` that touch *exactly* `contacts` 4-neighbours
        *within a single connected component* of that colour.
        Returns a single 2-D mask.
        """
        if contacts not in (1, 2, 3, 4):
            raise ValueError("contacts must be 1–4 for 4-connectivity")

        # 1. global mask of the colour, so we can blank out those cells at the end
        global_mask = GridSelector.colour(grid, colour)

        # 2. get each connected shape of that colour
        mask = GridSelector.colour(grid, colour)
        lbl, n = label(mask, structure=_FOUR_CONN)
        if n == 0:
            return np.zeros_like(global_mask, dtype=bool)

        # 3. for each component, count its 4-neighbours
        components = np.stack([(lbl == i) for i in range(1, n + 1)])
        counts = np.zeros_like(grid, dtype=int)
        
        # Count neighbors for each component
        for comp in components:
            # Use 2D convolution with proper axes
            neighbor_count = convolve(comp.astype(int), _FOUR_CONN, mode='constant', cval=0)
            counts += neighbor_count

        # 4. we want cells where the total count equals `contacts`
        hit = (counts == contacts)

        # 5. but exclude the coloured cells themselves
        return hit & ~global_mask

    # Create the contact functions using partial
    contact4_1, contact4_1.__name__ = partial(adjacent4, contacts=1), "contact4_1"
    contact4_2, contact4_2.__name__ = partial(adjacent4, contacts=2), "contact4_2"
    contact4_3, contact4_3.__name__ = partial(adjacent4, contacts=3), "contact4_3"
    contact4_4, contact4_4.__name__ = partial(adjacent4, contacts=4), "contact4_4"

    @staticmethod
    def adjacent8(grid: np.ndarray, colour: int, *,
                  contacts: int = 1) -> np.ndarray:
        """Same as :py:meth:`adjacent4` but with 8‑connectivity (1–8 contacts)."""
        if contacts not in range(1, 9):
            raise ValueError("contacts must be 1–8 for 8‑connectivity")

        mask = GridSelector.colour(grid, colour)
        cnt  = convolve(mask.astype(int), _EIGHT_CONN, mode="constant", cval=0)
        return (cnt == contacts) & ~mask
    
    contact8_1 = partial(adjacent8, contacts=1)
    contact8_2 = partial(adjacent8, contacts=2)
    contact8_3 = partial(adjacent8, contacts=3)
    contact8_4 = partial(adjacent8, contacts=4)
    contact8_5 = partial(adjacent8, contacts=5)
    contact8_6 = partial(adjacent8, contacts=6)
    contact8_7 = partial(adjacent8, contacts=7)
    contact8_8 = partial(adjacent8, contacts=8)

    # ─────────────────────────────────────────────── #
    # 5  Rectangles fully filled with given colour
    # ─────────────────────────────────────────────── #
    def rectangles(self, grid: np.ndarray, colour: int, *,
                   height: int, width: int) -> np.ndarray:
        """
        All H×W rectangles where **every** cell equals *colour*.
        Returns (num_rects, H, W) boolean masks.
        """
        H, W = grid.shape
        if not (self._min_geom <= height <= H and
                self._min_geom <= width  <= W):
            return np.zeros((0, H, W), dtype=bool)

        full_mask = GridSelector.colour(grid, colour)
        hits: list[np.ndarray] = []

        for y in range(H - height + 1):
            for x in range(W - width + 1):
                sub = full_mask[y:y + height, x:x + width]
                if np.all(sub):
                    m = np.zeros_like(full_mask, dtype=bool)
                    m[y:y + height, x:x + width] = True
                    hits.append(m)

        return np.stack(hits) if hits else np.zeros((0, H, W), dtype=bool)
