"""
test_selection_functions.py - Test script to verify selection functions output dimensions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dsl.select import GridSelector
from dsl.colour import ColorSelector

def test_selection_functions():
    # Create a test grid
    test_grid = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])
    
    # Initialize selectors
    grid_selector = GridSelector()
    color_selector = ColorSelector()
    
    # Test cases for each selection function
    test_cases = [
        # Basic selection functions
        ("all_cells", lambda: grid_selector.all_cells(test_grid)),
        ("colour", lambda: grid_selector.colour(test_grid, 1)),
        
        # Component-based functions
        ("components4", lambda: grid_selector.components4(test_grid, 1)),
        ("components8", lambda: grid_selector.components8(test_grid, 1)),
        ("independent_cells_4", lambda: grid_selector.independent_cells_4(test_grid, 1)),
        
        # Border functions
        ("outer_border4", lambda: grid_selector.outer_border4(test_grid, 1)),
        ("inner_border4", lambda: grid_selector.inner_border4(test_grid, 1)),
        ("outer_border8", lambda: grid_selector.outer_border8(test_grid, 1)),
        ("inner_border8", lambda: grid_selector.inner_border8(test_grid, 1)),
        ("grid_border", lambda: grid_selector.grid_border(test_grid, 1)),
        
        # Adjacency functions
        ("contact4_1", lambda: grid_selector.contact4_1(test_grid, 1)),
        ("contact4_2", lambda: grid_selector.contact4_2(test_grid, 1)),
        ("contact4_3", lambda: grid_selector.contact4_3(test_grid, 1)),
        ("contact4_4", lambda: grid_selector.contact4_4(test_grid, 1)),
    ]
    
    # Run tests
    print("Testing selection functions output dimensions:")
    print("-" * 50)
    
    for name, func in test_cases:
        try:
            result = func()
            if isinstance(result, np.ndarray):
                if result.ndim == 2:
                    print(f"✓ {name:15} - 2D output: shape {result.shape}")
                elif result.ndim == 3:
                    print(f"✓ {name:15} - 3D output: shape {result.shape}")
                else:
                    print(f"✗ {name:15} - Unexpected dimensions: {result.ndim}D")
            else:
                print(f"✗ {name:15} - Not a numpy array")
        except Exception as e:
            print(f"✗ {name:15} - Error: {str(e)}")
    
    print("\nSummary:")
    print("-" * 50)
    print("Expected behavior:")
    print("1. Most selection functions should return 2D boolean masks")
    print("2. Component-based functions (components4, components8) may return 3D arrays")
    print("3. All outputs should be boolean masks matching the input grid shape")

if __name__ == "__main__":
    test_selection_functions()