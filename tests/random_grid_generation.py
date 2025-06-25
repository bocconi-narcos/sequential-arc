"""
Test script for random grid generation and Poisson episode length sampling.
"""
import numpy as np
from dsl.utils.random_grid import generate_random_grid

NUM_GRIDS = 10
GRID_SHAPE = (5, 5)
NUM_COLORS = 4
LAMBDA_POISSON = 7
SEED = 42

def main():
    rng = np.random.default_rng(SEED)
    for i in range(NUM_GRIDS):
        grid = generate_random_grid(GRID_SHAPE, NUM_COLORS, rng)
        episode_length = rng.poisson(LAMBDA_POISSON)
        unique_colors = np.unique(grid)
        print(f"--- Grid {i+1} ---")
        print(grid)
        print(f"Unique colors: {unique_colors} (count: {len(unique_colors)})")
        print(f"Sampled episode length: {episode_length}")
        print()

if __name__ == "__main__":
    main() 