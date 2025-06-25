"""
Random grid generation utility for ARC-style tasks.

Provides a function to generate a random 2D grid with a specified shape and number of colors.
"""
import numpy as np
from typing import Tuple

def generate_random_grid(
    shape: Tuple[int, int],
    num_colors: int = 4,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Generate a random 2D grid with integer values in [0, num_colors-1].

    Args:
        shape: (H, W) tuple for grid size.
        num_colors: Number of unique colors to use (default: 4).
        rng: Optional numpy random Generator for reproducibility.

    Returns:
        grid: np.ndarray of shape (H, W) with random integers in [0, num_colors-1].
    """
    if rng is None:
        rng = np.random.default_rng()
    H, W = shape
    if num_colors < 1:
        raise ValueError("num_colors must be >= 1")
    return rng.integers(0, num_colors, size=(H, W), dtype=np.int8) 