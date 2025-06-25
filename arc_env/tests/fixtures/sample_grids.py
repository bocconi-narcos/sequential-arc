import numpy as np
from typing import Dict, List, Tuple

# This file provides sample ARC grids that can be used as fixtures in tests.
# These can represent inputs, outputs, or intermediate states for testing
# operations, solvers, or environment logic.

# --- Basic Grids ---
GRID_EMPTY_3x3 = np.zeros((3, 3), dtype=np.uint8)
GRID_FILLED_3x3_COLOR_1 = np.full((3, 3), 1, dtype=np.uint8)

GRID_DIAGONAL_3x3_COLOR_2 = np.array([
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 2]
], dtype=np.uint8)

GRID_checkerboard_2x2 = np.array([
    [1, 0],
    [0, 1]
], dtype=np.uint8)

# --- Grids from ARC Evaluation Set (Examples - Fictional or Simplified) ---
# It's good to use actual task examples if possible, but keep them small for unit tests.
# These would typically be loaded from JSON, but can be defined here for quick access.

# Example: A simplified version of a task input/output pair
# From a task like "fill the inner square"
TASK_FILL_INNER_SQUARE_INPUT_5x5 = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
], dtype=np.uint8) # Outer border color 1, inner is 0

TASK_FILL_INNER_SQUARE_OUTPUT_5x5 = np.array([
    [1, 1, 1, 1, 1],
    [1, 2, 2, 2, 1], # Inner filled with color 2
    [1, 2, 2, 2, 1],
    [1, 2, 2, 2, 1],
    [1, 1, 1, 1, 1]
], dtype=np.uint8)

# Example: A task involving moving a shape
TASK_MOVE_SHAPE_INPUT_4x4 = np.array([
    [3, 3, 0, 0],
    [3, 3, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
], dtype=np.uint8) # 2x2 shape of color 3 at top-left

TASK_MOVE_SHAPE_OUTPUT_4x4 = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 3, 3], # Shape moved to bottom-right
    [0, 0, 3, 3]
], dtype=np.uint8)


# --- Grid Masks for Selections ---
MASK_TOP_LEFT_2x2_ON_3x3 = np.array([
    [True,  True,  False],
    [True,  True,  False],
    [False, False, False]
], dtype=bool)

MASK_CENTER_PIXEL_3x3 = np.array([
    [False, False, False],
    [False, True,  False],
    [False, False, False]
], dtype=bool)

# --- Collection of all sample grids (optional, for easy iteration if needed) ---
ALL_SAMPLE_GRIDS: Dict[str, np.ndarray] = {
    "empty_3x3": GRID_EMPTY_3x3,
    "filled_3x3_color_1": GRID_FILLED_3x3_COLOR_1,
    "diagonal_3x3_color_2": GRID_DIAGONAL_3x3_COLOR_2,
    "checkerboard_2x2": GRID_checkerboard_2x2,
    "task_fill_inner_input_5x5": TASK_FILL_INNER_SQUARE_INPUT_5x5,
    "task_fill_inner_output_5x5": TASK_FILL_INNER_SQUARE_OUTPUT_5x5,
    "task_move_shape_input_4x4": TASK_MOVE_SHAPE_INPUT_4x4,
    "task_move_shape_output_4x4": TASK_MOVE_SHAPE_OUTPUT_4x4,
}

ALL_SAMPLE_MASKS: Dict[str, np.ndarray] = {
    "top_left_2x2_on_3x3": MASK_TOP_LEFT_2x2_ON_3x3,
    "center_pixel_3x3": MASK_CENTER_PIXEL_3x3,
}


# Helper function to get a copy of a sample grid to avoid in-place modification issues
def get_sample_grid(name: str) -> Optional[np.ndarray]:
    """Returns a copy of a named sample grid, or None if not found."""
    if name in ALL_SAMPLE_GRIDS:
        return np.copy(ALL_SAMPLE_GRIDS[name])
    return None

def get_sample_mask(name: str) -> Optional[np.ndarray]:
    """Returns a copy of a named sample mask, or None if not found."""
    if name in ALL_SAMPLE_MASKS:
        return np.copy(ALL_SAMPLE_MASKS[name])
    return None


if __name__ == '__main__':
    print("Sample Grids Available:")
    for name, grid_array in ALL_SAMPLE_GRIDS.items():
        print(f"\nGrid: {name} (shape: {grid_array.shape})")
        print(grid_array)

    print("\nSample Masks Available:")
    for name, mask_array in ALL_SAMPLE_MASKS.items():
        print(f"\nMask: {name} (shape: {mask_array.shape})")
        print(mask_array.astype(np.uint8)) # Print bool as 0/1

    # Example of getting a copy
    copied_grid = get_sample_grid("diagonal_3x3_color_2")
    if copied_grid is not None:
        # Modify the copy, original in ALL_SAMPLE_GRIDS should remain unchanged
        copied_grid[0,0] = 99
        # print("\nOriginal diagonal_3x3_color_2:")
        # print(ALL_SAMPLE_GRIDS["diagonal_3x3_color_2"])
        # print("Modified copy:")
        # print(copied_grid)
        assert ALL_SAMPLE_GRIDS["diagonal_3x3_color_2"][0,0] == 2 # Original unchanged
    else:
        print("Error: Could not retrieve 'diagonal_3x3_color_2' sample.")
