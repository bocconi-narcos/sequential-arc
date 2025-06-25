from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional, Dict

# This module will contain utility functions for common grid manipulations
# and analyses relevant to ARC tasks and DSL operations.

def get_grid_shape(grid: np.ndarray) -> Tuple[int, int]:
    """Returns the shape (height, width) of the grid."""
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        raise ValueError("Input grid must be a 2D numpy array.")
    return grid.shape

def get_pixel(grid: np.ndarray, row: int, col: int) -> Optional[int]:
    """
    Safely retrieves the pixel value at (row, col).
    Returns None if coordinates are out of bounds.
    """
    height, width = grid.shape
    if 0 <= row < height and 0 <= col < width:
        return int(grid[row, col])
    return None

def set_pixel(grid: np.ndarray, row: int, col: int, color: int) -> bool:
    """
    Safely sets the pixel value at (row, col) if coordinates are valid.
    Modifies the grid in-place.
    Returns True if successful, False otherwise.
    """
    height, width = grid.shape
    if 0 <= row < height and 0 <= col < width:
        grid[row, col] = color
        return True
    return False

def find_bounding_box(grid: np.ndarray, mask: Optional[np.ndarray] = None, ignore_color: Optional[int] = None) -> Optional[Tuple[int, int, int, int]]:
    """
    Finds the bounding box (r_min, c_min, r_max, c_max) of non-ignored elements
    within the grid or a given mask.
    If a mask is provided, only considers True elements in the mask.
    If ignore_color is provided, pixels of this color are treated as background.
    Returns None if the grid/mask is empty or only contains ignored elements.
    """
    target_grid = grid
    if mask is not None:
        if grid.shape != mask.shape:
            raise ValueError("Grid and mask must have the same shape.")
        # Consider only the part of the grid where mask is True
        # For finding coordinates, it's easier to work with indices from the original grid
        rows, cols = np.where(mask)
        if not rows.size: return None # Empty mask
    else:
        # If no mask, consider all pixels not matching ignore_color
        if ignore_color is not None:
            rows, cols = np.where(target_grid != ignore_color)
        else:
            # If no mask and no ignore_color, all pixels are candidates (unless grid is empty)
            if target_grid.size == 0: return None
            rows, cols = np.indices(target_grid.shape) # Gets all indices
            rows, cols = rows.flatten(), cols.flatten() # Ensure they are 1D arrays for min/max

    if not rows.size: return None # No relevant pixels found

    r_min, r_max = np.min(rows), np.max(rows)
    c_min, c_max = np.min(cols), np.max(cols)

    return (r_min, c_min, r_max, c_max)


def extract_subgrid(grid: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Extracts a subgrid defined by a bounding box (r_min, c_min, r_max, c_max).
    """
    r_min, c_min, r_max, c_max = bbox
    return grid[r_min : r_max + 1, c_min : c_max + 1]

def
place_subgrid(
    main_grid: np.ndarray,
    subgrid: np.ndarray,
    row: int,
    col: int,
    ignore_color_in_subgrid: Optional[int] = None,
    mask_on_subgrid: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Places a subgrid onto the main_grid at the specified (row, col) top-left position.
    Modifies main_grid in-place. Returns the modified main_grid.

    Args:
        main_grid: The target grid.
        subgrid: The smaller grid to place.
        row, col: Top-left coordinates on main_grid where subgrid will be placed.
        ignore_color_in_subgrid: If specified, pixels of this color in subgrid are not copied (transparent).
        mask_on_subgrid: A boolean mask of the same shape as subgrid. Only True parts of subgrid are placed.
                         If both ignore_color and mask_on_subgrid are given, mask_on_subgrid takes precedence
                         for determining which pixels *can* be copied, then ignore_color filters those.
    """
    sg_h, sg_w = subgrid.shape
    mg_h, mg_w = main_grid.shape

    for r_offset in range(sg_h):
        for c_offset in range(sg_w):
            # Check if this pixel in subgrid should be copied based on mask_on_subgrid
            if mask_on_subgrid is not None:
                if not mask_on_subgrid[r_offset, c_offset]:
                    continue # Skip if not in subgrid's mask

            pixel_value = subgrid[r_offset, c_offset]

            # Check for transparency
            if ignore_color_in_subgrid is not None and pixel_value == ignore_color_in_subgrid:
                continue

            # Determine target coordinates on main grid
            target_r, target_c = row + r_offset, col + c_offset

            # Place if within bounds of main_grid
            if 0 <= target_r < mg_h and 0 <= target_c < mg_w:
                main_grid[target_r, target_c] = pixel_value
    return main_grid


def find_objects(grid: np.ndarray, connectivity: int = 1, ignore_color: Optional[int] = None) -> List[np.ndarray]:
    """
    Finds connected components (objects) in the grid.
    Uses skimage.measure.label if available, otherwise a basic implementation.

    Args:
        grid: The input grid.
        connectivity: 1 for 4-connectivity, 2 for 8-connectivity.
        ignore_color: Pixels of this color are treated as background.

    Returns:
        A list of boolean masks, each mask representing one object.
    """
    try:
        from skimage.measure import label as skimage_label

        if ignore_color is not None:
            binary_grid = (grid != ignore_color).astype(int)
        else:
            # If no ignore_color, consider all non-zero pixels as foreground,
            # or define objects by changes in color.
            # For simplicity, let's assume 0 is background if ignore_color is None.
            binary_grid = (grid != 0).astype(int)
            # More advanced: label different colors separately or all non-background.

        labeled_grid, num_labels = skimage_label(
            binary_grid,
            connectivity=connectivity, # 1 for 4-way, 2 for 8-way
            return_num=True,
            background=0 # Pixels with value 0 in binary_grid are background
        )

        object_masks = []
        for i in range(1, num_labels + 1): # Labels are 1-indexed
            object_masks.append(labeled_grid == i)
        return object_masks

    except ImportError:
        print("Warning: skimage.measure.label not found. Object detection will be basic or unavailable.")
        # Placeholder for a very basic object finding if skimage is not present.
        # This would be significantly more complex to implement robustly here.
        # For now, return an empty list or a single mask of non-ignored pixels.
        if ignore_color is not None:
            return [(grid != ignore_color)] if np.any(grid != ignore_color) else []
        else: # Assume 0 is background
             return [(grid != 0)] if np.any(grid != 0) else []


# More utilities to be added:
# - get_neighbors(grid, r, c, connectivity=1)
# - count_colors(grid, mask=None) -> Dict[int, int]
# - is_grid_symmetric(grid, axis='horizontal'|'vertical'|'diagonal_tl_br'|'diagonal_tr_bl')
# - crop_to_content(grid, ignore_color=0, padding=0) -> np.ndarray
# - ... and many more based on ARC problem characteristics.

# Example usage
if __name__ == '__main__':
    test_grid = np.array([
        [0, 1, 1, 0],
        [0, 1, 0, 2],
        [3, 0, 0, 2],
        [0, 0, 0, 0]
    ])
    print("Test Grid:\n", test_grid)
    print("Shape:", get_grid_shape(test_grid))
    print("Pixel (1,1):", get_pixel(test_grid, 1, 1)) # Should be 1
    print("Pixel (5,5):", get_pixel(test_grid, 5, 5)) # Should be None

    # Bounding box of non-zero elements
    bbox_all_content = find_bounding_box(test_grid, ignore_color=0)
    print("BBox of non-zero content:", bbox_all_content) # Expected: (0, 1, 2, 3)

    if bbox_all_content:
        sub = extract_subgrid(test_grid, bbox_all_content)
        print("Extracted subgrid (non-zero content):\n", sub)

    # Bounding box for color 1
    mask_color_1 = (test_grid == 1)
    bbox_color_1 = find_bounding_box(test_grid, mask=mask_color_1)
    print("BBox of color 1:", bbox_color_1) # Expected: (0, 1, 1, 2)
    if bbox_color_1:
        sub_color_1 = extract_subgrid(test_grid, bbox_color_1)
        print("Extracted subgrid (color 1):\n", sub_color_1) # [[1,1],[1,0]]

    # Place subgrid
    grid_to_place_on = np.zeros((5,5), dtype=int)
    sub_to_place = np.array([[9,9],[9,9]])
    print("Placing subgrid:\n", sub_to_place)
    print("Onto (before):\n", grid_to_place_on)
    place_subgrid(grid_to_place_on, sub_to_place, 1, 1, ignore_color_in_subgrid=None)
    print("Onto (after):\n", grid_to_place_on)

    # Find objects
    print("\nFinding objects in test_grid (ignore_color=0):")
    objects = find_objects(test_grid, ignore_color=0)
    if objects:
        for i, obj_mask in enumerate(objects):
            print(f"Object {i+1} mask:\n", obj_mask.astype(int))
            obj_content = np.zeros_like(test_grid)
            obj_content[obj_mask] = test_grid[obj_mask]
            print(f"Object {i+1} content:\n", obj_content)
    else:
        print("No objects found (or skimage not available for advanced detection).")

    # Test find_objects with a grid that has distinct objects
    obj_test_grid = np.array([
        [1,1,0,2,2],
        [1,0,0,0,2],
        [0,0,3,0,0],
        [4,4,0,0,0]
    ])
    print("\nFinding objects in obj_test_grid (ignore_color=0):")
    objects_2 = find_objects(obj_test_grid, ignore_color=0)
    if objects_2:
        for i, obj_mask in enumerate(objects_2):
            print(f"Object {i+1} mask:\n", obj_mask.astype(int))
    else:
        print("No objects found in obj_test_grid.")
