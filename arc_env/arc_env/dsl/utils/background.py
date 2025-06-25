from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List # Added List for ignore_colors type hint
from collections import Counter # For get_border_color's Counter usage.

from .color_utils import count_colors, ARC_BLACK

# Utilities for detecting or inferring background properties of an ARC grid.

def get_dominant_color(grid: np.ndarray, ignore_colors: Optional[list[int]] = None) -> Optional[int]:
    """
    Determines the most dominant color in the grid, potentially ignoring some.
    This is often a good candidate for the background color.
    """
    color_counts = count_colors(grid)

    if ignore_colors:
        for ignored_c in ignore_colors:
            if ignored_c in color_counts:
                del color_counts[ignored_c]

    if not color_counts:
        return None

    return color_counts.most_common(1)[0][0]


def get_border_color(grid: np.ndarray, border_width: int = 1) -> Optional[int]:
    """
    Determines the most frequent color along the borders of the grid.
    Another common heuristic for background color.

    Args:
        grid: The input grid.
        border_width: How many pixels deep to consider as the border.

    Returns:
        The most frequent color in the border region, or None if grid is too small.
    """
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        raise ValueError("Input grid must be a 2D numpy array.")

    h, w = grid.shape
    if h < 2 * border_width or w < 2 * border_width:
        # Grid is too small for the specified border width, fallback to dominant color of whole grid?
        # Or return None, or dominant of what border exists.
        # For now, let's consider what border exists.
        if h == 0 or w == 0: return None # Empty grid

        # Collect all border pixels possible
        border_pixels = []
        # Top rows
        border_pixels.extend(grid[0:min(border_width, h), :].flatten())
        # Bottom rows (avoid double counting if h < 2*border_width)
        if h > border_width:
            border_pixels.extend(grid[max(h - border_width, border_width):h, :].flatten())
        # Left columns (avoiding corners already counted)
        start_row_for_sides = min(border_width, h)
        end_row_for_sides = max(h - border_width, 0) # should be max(h-border_width, border_width) if h > border_width
        if h <= 2*border_width : start_row_for_sides = 0 # effectively, already counted by top/bottom if too thin

        if w > 0 and end_row_for_sides > start_row_for_sides : # only if there are central side parts left
             border_pixels.extend(grid[start_row_for_sides:end_row_for_sides, 0:min(border_width, w)].flatten())
             # Right columns (avoiding corners)
             if w > border_width:
                 border_pixels.extend(grid[start_row_for_sides:end_row_for_sides, max(w - border_width, border_width):w].flatten())

        if not border_pixels: # e.g. 1x1 grid with border_width=1
            return grid[0,0] if grid.size > 0 else None

        counts = Counter(border_pixels)
        return counts.most_common(1)[0][0] if counts else None


    # Simplified border extraction for valid border_width
    border_mask = np.zeros_like(grid, dtype=bool)
    border_mask[:border_width, :] = True  # Top border
    border_mask[-border_width:, :] = True # Bottom border
    border_mask[:, :border_width] = True  # Left border
    border_mask[:, -border_width:] = True # Right border

    border_colors = grid[border_mask]
    if border_colors.size == 0:
        return None # Should not happen if previous checks pass

    counts = Counter(border_colors)
    return counts.most_common(1)[0][0]


def infer_background_color(
    grid: np.ndarray,
    default_candidate: int = ARC_BLACK,
    use_dominant: bool = True,
    use_border: bool = True,
    border_check_width: int = 1
) -> int:
    """
    Infers the background color of an ARC grid using heuristics.

    Heuristics used:
    1. If `use_dominant` is True, considers the overall dominant color.
    2. If `use_border` is True, considers the dominant color of the border.
    3. If both are used, might prefer border color or the one that's more common if they differ.
       Currently, it prioritizes border color if available and different from overall dominant.
       If only one heuristic is enabled, it uses that.
       If both are enabled and yield same result, uses that.
       If results differ or a heuristic yields None, it might fall back.
       If all heuristics fail or are disabled, returns `default_candidate`.

    Args:
        grid: The input grid.
        default_candidate: The color to return if heuristics fail or are inconclusive.
        use_dominant: Whether to use the dominant color heuristic.
        use_border: Whether to use the border color heuristic.
        border_check_width: Width of the border to check for border color heuristic.

    Returns:
        The inferred background color (integer).
    """
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        raise ValueError("Input grid must be a 2D numpy array.")
    if grid.size == 0:
        return default_candidate # Empty grid

    dominant_bg: Optional[int] = None
    border_bg: Optional[int] = None

    if use_dominant:
        dominant_bg = get_dominant_color(grid)

    if use_border:
        border_bg = get_border_color(grid, border_width=border_check_width)

    # Decision logic:
    if use_border and border_bg is not None:
        if use_dominant and dominant_bg is not None:
            # Both available. Is there a preference?
            # Often, border color is a strong indicator.
            # If border_bg is a very minor color overall, dominant_bg might be better.
            # For now, simple preference: if border_bg seems valid, use it.
            # One simple check: if border_bg is also the dominant_bg, great.
            if border_bg == dominant_bg:
                return border_bg
            # If they differ, let's prefer border_bg if it's not extremely rare.
            # This logic can be refined. For now, let's just prioritize border_bg if available.
            return border_bg
        else: # Only border_bg available (or dominant disabled)
            return border_bg
    elif use_dominant and dominant_bg is not None: # Only dominant_bg available (or border disabled/failed)
        return dominant_bg

    return default_candidate


def create_background_mask(grid: np.ndarray, background_color: Optional[int] = None) -> np.ndarray:
    """
    Creates a boolean mask where True indicates a background pixel.
    If background_color is not provided, it's inferred.
    """
    if background_color is None:
        background_color = infer_background_color(grid)

    return grid == background_color


# Example Usage
if __name__ == '__main__':
    from collections import Counter # Import here for example usage if not at top level

    grid1 = np.array([ # Black background, dominant and border
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,1,0,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
    ])
    grid2 = np.array([ # Grey border, black inner dominant (more black pixels overall)
        [5,5,5,5,5],
        [5,0,0,0,5],
        [5,0,1,0,5],
        [5,0,0,0,5],
        [5,5,5,5,5],
    ])
    grid3 = np.array([ # Mixed border, but mostly 0. Overall dominant is 2.
        [0,2,2,2,0],
        [2,2,2,2,2],
        [2,2,0,2,2],
        [2,2,2,2,2],
        [0,2,2,2,5],
    ])
    grid4_small = np.array([[1,1],[1,0]])

    print("Grid 1 (black bg):\n", grid1)
    print(f"Dominant color: {get_dominant_color(grid1)}") # 0
    print(f"Border color: {get_border_color(grid1)}")     # 0
    print(f"Inferred background: {infer_background_color(grid1)}") # 0
    print(f"Background mask:\n{create_background_mask(grid1).astype(int)}")

    print("\nGrid 2 (grey border, black inner dominant):\n", grid2)
    print(f"Dominant color: {get_dominant_color(grid2)}") # 0 (9 zeros vs 16 fives)
    print(f"Border color: {get_border_color(grid2)}")     # 5
    print(f"Inferred background: {infer_background_color(grid2)}") # Should prefer 5 (border)

    print("\nGrid 3 (mixed border, 0 is most common on border, 2 is dominant overall):\n", grid3)
    print(f"Dominant color: {get_dominant_color(grid3)}") # 2
    print(f"Border color: {get_border_color(grid3)}")     # 0
    print(f"Inferred background: {infer_background_color(grid3)}") # Should prefer 0 (border)

    print("\nGrid 4 (small):\n", grid4_small)
    print(f"Dominant color: {get_dominant_color(grid4_small)}") # 1
    print(f"Border color: {get_border_color(grid4_small)}")     # 1 (border is whole grid)
    print(f"Inferred background: {infer_background_color(grid4_small, default_candidate=ARC_BLACK)}") # 1

    grid_all_same = np.full((3,3), 2)
    print("\nGrid all same color (2):\n", grid_all_same)
    print(f"Dominant color: {get_dominant_color(grid_all_same)}") # 2
    print(f"Border color: {get_border_color(grid_all_same)}")     # 2
    print(f"Inferred background: {infer_background_color(grid_all_same)}") # 2

    empty_grid = np.array([[]])
    print("\nEmpty Grid:\n", empty_grid)
    # print(f"Dominant color: {get_dominant_color(empty_grid)}") # Error or None
    # print(f"Border color: {get_border_color(empty_grid)}")     # Error or None
    print(f"Inferred background: {infer_background_color(empty_grid)}") # default_candidate (0)
