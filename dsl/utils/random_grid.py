"""
Random grid generation utility for ARC-style tasks.

Provides a function to generate a random 2D grid with a specified shape and number of colors.
Colors are constrained to the ARC palette range [0, 9].
"""
import numpy as np
from typing import Tuple

import numpy as np
from typing import Tuple, Optional

def generate_random_grid(
    shape: Tuple[int, int],
    num_colors: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a random 2D grid whose entries are drawn from a
    random subset of `num_colors` distinct values in [0,9].

    Args:
        shape: (H, W) tuple for grid size.
        num_colors: How many distinct colors to pick from 0–9. 
                    Must be between 1 and 10.
        rng: Optional numpy random Generator for reproducibility.

    Returns:
        grid: np.ndarray of shape (H, W), dtype=int8, whose values
              are in the randomly chosen subset.
    """
    H, W = shape
    if not (1 <= num_colors <= 10):
        raise ValueError("num_colors must be between 1 and 10")

    if rng is None:
        rng = np.random.default_rng()

    # pick `num_colors` distinct palette values from 0…9
    palette = rng.choice(np.arange(10), size=num_colors, replace=False)

    # now fill the grid by sampling from that palette
    grid = rng.choice(palette, size=(H, W)).astype(np.int8)

    # Assert that the grid values are within 0 and 9
    if not np.all((grid >= 0) & (grid <= 9)):
        raise ValueError(f"Grid values must be in range [0, 9], found min={grid.min()}, max={grid.max()}")
    return grid

if __name__ == "__main__":
    # Example usage
    grid = generate_random_grid((5, 5), num_colors=4)
    print("Generated random grid:")
    print(grid)
    print("Unique colors in grid:", np.unique(grid))