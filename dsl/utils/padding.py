"""
Utility helpers for zero‑copy padding / unpadding of 2‑D grids.

* ``pad_grid``   – embed a grid into a larger canvas, filling the rest with
                   ``fill_val`` (default −1).
* ``unpad_grid`` – strip all‑``fill_val`` rows / cols around the content.

Both functions are *pure* (no side effects) and agnostic of the environment;
the caller supplies the desired canvas shape.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def pad_grid(grid: np.ndarray,
             target_shape: Tuple[int, int],
             *,
             fill_val: int = -1) -> np.ndarray:
    """
    Return a new array of shape ``target_shape`` where ``grid`` is placed in
    the top‑left corner and the rest is filled with ``fill_val``.
    """
    H, W = target_shape
    h, w = grid.shape
    if h > H or w > W:
        raise ValueError(f"Grid {grid.shape} larger than canvas {target_shape}")
    canvas = np.full(target_shape, fill_val, dtype=grid.dtype)
    canvas[:h, :w] = grid
    return canvas


def unpad_grid(grid: np.ndarray, *, fill_val: int = -1) -> np.ndarray:
    """
    Remove rows/cols that consist **entirely** of ``fill_val``.
    Returns a *view* when possible, otherwise a copy.
    """
    keep_rows = ~(np.all(grid == fill_val, axis=1))
    keep_cols = ~(np.all(grid == fill_val, axis=0))
    if not keep_rows.any() or not keep_cols.any():
        return np.empty((0, 0), dtype=grid.dtype)
    return grid[np.ix_(keep_rows, keep_cols)]
