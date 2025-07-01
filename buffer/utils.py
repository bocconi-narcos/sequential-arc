import os
import pickle
import numpy as np
import random
import json
from collections import deque
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import yaml
import h5py

from env import ARCEnv
from action_space import ARCActionSpace


def load_and_filter_grids(solutions_data, challenges_data, max_h, max_w):
    """
    Filters grids from solutions_data and challenges_data based on maximum height and width.
    Returns a list of (task_key, example_index) for valid grids.
    """
    valid_grids = []
    
    for task_key in challenges_data:
        if 'train' in challenges_data[task_key]:
            for i, example in enumerate(challenges_data[task_key]['train']):
                try:
                    # Check both input and output dimensions
                    input_grid = np.array(example['input'])
                    output_grid = np.array(solutions_data[task_key])  # Solution is directly in solutions_data
                    
                    if (input_grid.shape[0] <= max_h and input_grid.shape[1] <= max_w and
                        output_grid.shape[0] <= max_h and output_grid.shape[1] <= max_w):
                        valid_grids.append((task_key, i))
                except Exception as e:
                    print(f"Warning: Could not parse grid for task {task_key}, example {i}: {e}")
                    continue
    
    return valid_grids

def process_state(state: np.ndarray) -> np.ndarray:
    """
    Process a state array to ensure it's 2D. If it's 3D (from selection functions),
    combine all layers into a single 2D array.
    
    Args:
        state: Input state array (2D or 3D)
        
    Returns:
        2D numpy array
    """
    if state.ndim == 3:
        # If we have multiple layers (from selection functions),
        # combine them by taking the maximum value across layers
        return np.max(state, axis=0)
    return state

def count_unique_colors(grid: np.ndarray) -> int:
    """Return the number of unique colors in the grid."""
    return len(np.unique(grid))


def get_selection_mask(action_space: ARCActionSpace, action: Dict[str, int], grid: np.ndarray) -> np.ndarray:
    """
    Given an action and a grid, return the boolean selection mask for the action.
    """
    _, selection_fn, _ = action_space.decode(action)
    # Some selectors require a color argument, others do not
    try:
        mask = selection_fn(grid, action["colour"])
    except TypeError:
        mask = selection_fn(grid)
    # Ensure mask is 2D boolean
    if mask.ndim == 3:
        mask = np.any(mask, axis=0)
    return mask.astype(bool)


def validate_colors(grid: np.ndarray, name: str = "grid") -> None:
    """
    Validate that all colors in a grid are within the valid ARC range [0, 9].
    """
    if grid.min() < 0 or grid.max() > 9:
        raise ValueError(f"Colors in {name} are outside valid ARC range [0, 9]. "
                        f"Found range [{grid.min()}, {grid.max()}]")


def get_grid_shape(grid: np.ndarray) -> Tuple[int, int]:
    """
    Return the shape (height, width) of the grid.
    """
    return tuple(grid.shape)


def count_unique_colors_exclude_padding(grid: np.ndarray, padding_val: int = -1) -> int:
    """
    Return the number of unique colors in the grid, excluding the padding value.
    """
    unique = np.unique(grid)
    unique_no_pad = unique[unique != padding_val]
    return len(unique_no_pad)


def most_present_color_exclude_padding(grid: np.ndarray, padding_val: int = -1) -> int:
    """
    Return the color value that appears most frequently in the grid, excluding the padding value.
    If the grid is empty or only contains padding, return -1.
    """
    flat = grid.flatten()
    flat_no_pad = flat[flat != padding_val]
    if flat_no_pad.size == 0:
        return -1
    vals, counts = np.unique(flat_no_pad, return_counts=True)
    return int(vals[np.argmax(counts)])

def least_present_color_exclude_padding(grid: np.ndarray, padding_val: int = -1) -> int:
    """
    Return the color value that appears least frequently in the grid, excluding the padding value.
    If the grid is empty or only contains padding, return -1.
    """
    flat = grid.flatten()
    flat_no_pad = flat[flat != padding_val]
    if flat_no_pad.size == 0:
        return -1
    vals, counts = np.unique(flat_no_pad, return_counts=True)
    return int(vals[np.argmin(counts)])

def validate_grid_padding(padded_grid: np.ndarray, original_shape: tuple, canvas_size: int, grid_name: str):
    """
    Validate that a padded grid has correct padding structure:
    - Padded regions (outside original_shape) should contain only -1s
    - Non-padded regions (within original_shape) should not contain any -1s
    
    Args:
        padded_grid: The padded grid to validate
        original_shape: (height, width) of the original unpadded grid
        canvas_size: Size of the padded canvas
        grid_name: Name of the grid for error messages
    """
    orig_h, orig_w = original_shape
    
    # Check that non-padded region doesn't contain -1s
    non_padded_region = padded_grid[:orig_h, :orig_w]
    if np.any(non_padded_region == -1):
        raise AssertionError(f"Non-padded region of {grid_name} contains -1s, which should only appear in padding")
    
    # Check that padded regions contain only -1s
    # Right padding (if exists)
    if orig_w < canvas_size:
        right_padding = padded_grid[:orig_h, orig_w:]
        if not np.all(right_padding == -1):
            raise AssertionError(f"Right padding of {grid_name} contains non-(-1) values")
    
    # Bottom padding (if exists)
    if orig_h < canvas_size:
        bottom_padding = padded_grid[orig_h:, :]
        if not np.all(bottom_padding == -1):
            raise AssertionError(f"Bottom padding of {grid_name} contains non-(-1) values")
