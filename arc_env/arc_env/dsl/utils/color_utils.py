from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Union, Counter as TypingCounter # Counter from typing for type hints
from collections import Counter # Actual Counter implementation

# ARC colors are typically integers from 0 to 9.
# 0 is often black / background.

ARC_BLACK = 0
ARC_BLUE = 1
ARC_RED = 2
ARC_GREEN = 3
ARC_YELLOW = 4
ARC_GREY = 5 # Or gray
ARC_MAGENTA = 6 # Or fuchsia, pink
ARC_ORANGE = 7
ARC_CYAN = 8
ARC_BROWN = 9 # Or dark red, etc.

# Mapping for convenience, can be expanded or made more formal
ARC_COLOR_MAP: Dict[str, int] = {
    "black": ARC_BLACK,
    "blue": ARC_BLUE,
    "red": ARC_RED,
    "green": ARC_GREEN,
    "yellow": ARC_YELLOW,
    "grey": ARC_GREY, "gray": ARC_GREY,
    "magenta": ARC_MAGENTA, "fuchsia": ARC_MAGENTA, "pink": ARC_MAGENTA,
    "orange": ARC_ORANGE,
    "cyan": ARC_CYAN,
    "brown": ARC_BROWN,
}
ARC_COLOR_NAME_MAP: Dict[int, str] = {v: k for k, v in ARC_COLOR_MAP.items() if k not in ["gray", "fuchsia", "pink"]} # Simple reverse map

def get_color_value(color: Union[str, int]) -> int:
    """Converts a color name or int to its integer value. Raises ValueError if invalid."""
    if isinstance(color, str):
        color_val = ARC_COLOR_MAP.get(color.lower())
        if color_val is None:
            raise ValueError(f"Unknown color name: {color}. Valid names: {list(ARC_COLOR_MAP.keys())}")
        return color_val
    elif isinstance(color, int):
        if not (0 <= color <= 9): # Basic ARC range check
            # Could be more flexible if more colors are allowed by a specific env config
            raise ValueError(f"Invalid color integer: {color}. Must be between 0 and 9 for standard ARC.")
        return color
    else:
        raise TypeError(f"Color must be a string or an integer, got {type(color).__name__}.")

def get_color_name(color_value: int) -> str:
    """Returns a canonical name for a color value. Returns 'unknown' if not in map."""
    return ARC_COLOR_NAME_MAP.get(color_value, "unknown")


def count_colors(grid: np.ndarray, mask: Optional[np.ndarray] = None) -> TypingCounter[int]:
    """
    Counts the occurrences of each color in the grid or within a masked area.

    Args:
        grid: The input grid.
        mask: Optional boolean mask. If provided, only count colors where mask is True.
              If None, count colors in the entire grid.

    Returns:
        A Counter object mapping color values to their counts.
    """
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        raise ValueError("Input grid must be a 2D numpy array.")

    if mask is not None:
        if grid.shape != mask.shape:
            raise ValueError("Grid and mask must have the same shape.")
        if mask.dtype != bool:
            raise ValueError("Mask must be a boolean numpy array.")
        pixels_to_count = grid[mask]
    else:
        pixels_to_count = grid.flatten()

    return Counter(pixels_to_count)

def get_most_frequent_color(grid: np.ndarray, mask: Optional[np.ndarray] = None, ignore_colors: Optional[List[int]] = None) -> Optional[int]:
    """
    Finds the most frequent color in the grid/mask, optionally ignoring some colors.
    Returns None if no suitable colors are found (e.g., all ignored or empty selection).
    If there's a tie, returns one of the most frequent ones (behavior may vary).
    """
    color_counts = count_colors(grid, mask)

    if ignore_colors:
        for ignored_c in ignore_colors:
            if ignored_c in color_counts:
                del color_counts[ignored_c]

    if not color_counts:
        return None

    # most_common() returns a list of (element, count) tuples.
    # We want the element (color) from the first tuple.
    return color_counts.most_common(1)[0][0]

def get_least_frequent_color(grid: np.ndarray, mask: Optional[np.ndarray] = None, ignore_colors: Optional[List[int]] = None) -> Optional[int]:
    """
    Finds the least frequent non-ignored color.
    Returns None if no suitable colors are found.
    If there's a tie, returns one of the least frequent.
    """
    color_counts = count_colors(grid, mask)

    if ignore_colors:
        for ignored_c in ignore_colors:
            if ignored_c in color_counts:
                del color_counts[ignored_c]

    if not color_counts:
        return None

    # Get all items, sort by count (ascending), then pick the first one's color
    # Counter.items() gives dict_items, which can be sorted.
    # Or, use most_common() and pick the last one.
    # most_common() with no args sorts from most to least.
    least_common_list = color_counts.most_common()
    if not least_common_list: # Should be redundant if color_counts was not empty
        return None
    return least_common_list[-1][0]


def get_unique_colors(grid: np.ndarray, mask: Optional[np.ndarray] = None) -> List[int]:
    """
    Returns a sorted list of unique colors present in the grid or masked area.
    """
    color_counts = count_colors(grid, mask)
    return sorted(list(color_counts.keys()))


# Example Usage:
if __name__ == '__main__':
    test_grid = np.array([
        [ARC_BLACK, ARC_BLUE,  ARC_BLUE,  ARC_RED],
        [ARC_BLACK, ARC_BLUE,  ARC_GREEN, ARC_RED],
        [ARC_GREY,  ARC_BLACK, ARC_GREEN, ARC_RED],
    ])
    print("Test Grid:\n", test_grid)

    print(f"\nColor value for 'red': {get_color_value('red')}")
    print(f"Color value for ARC_GREEN ({ARC_GREEN}): {get_color_value(ARC_GREEN)}")
    try:
        get_color_value("purple")
    except ValueError as e:
        print(f"Error (expected): {e}")
    try:
        get_color_value(10)
    except ValueError as e:
        print(f"Error (expected): {e}")

    print(f"\nColor name for {ARC_BLUE}: {get_color_name(ARC_BLUE)}")
    print(f"Color name for 99: {get_color_name(99)}")


    all_counts = count_colors(test_grid)
    print(f"\nAll color counts: {all_counts}")
    # Expected: Counter({0: 3, 1: 3, 2: 3, 3: 2, 5: 1}) (ARC_BLACK:3, ARC_BLUE:3, ARC_RED:3, ARC_GREEN:2, ARC_GREY:1)

    test_mask = np.array([
        [True,  True,  False, False],
        [True,  True,  False, False],
        [False, False, False, False]
    ], dtype=bool) # Selects top-left 2x2 area: [[0,1],[0,1]]

    masked_counts = count_colors(test_grid, mask=test_mask)
    print(f"Masked color counts (top-left 2x2): {masked_counts}")
    # Expected: Counter({0: 2, 1: 2})

    print(f"\nMost frequent color in grid: {get_most_frequent_color(test_grid)}") # Tie, e.g. 0, 1, or 2
    print(f"Most frequent (ignore black {ARC_BLACK}): {get_most_frequent_color(test_grid, ignore_colors=[ARC_BLACK])}") # Tie, e.g. 1 or 2
    print(f"Most frequent in mask: {get_most_frequent_color(test_grid, mask=test_mask)}") # Tie, 0 or 1

    print(f"\nLeast frequent color in grid: {get_least_frequent_color(test_grid)}") # 5 (ARC_GREY)
    print(f"Least frequent (ignore grey {ARC_GREY}): {get_least_frequent_color(test_grid, ignore_colors=[ARC_GREY])}") # 3 (ARC_GREEN)

    empty_mask = np.zeros_like(test_grid, dtype=bool)
    print(f"Most frequent in empty mask: {get_most_frequent_color(test_grid, mask=empty_mask)}") # None

    print(f"\nUnique colors in grid: {get_unique_colors(test_grid)}") # [0,1,2,3,5]
    print(f"Unique colors in mask: {get_unique_colors(test_grid, mask=test_mask)}") # [0,1]
