"""
tests/visualize_selection.py - Interactive script to visualize selection function outputs
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsl.select import GridSelector
from dsl.colour import ColorSelector

def create_test_grid(size=5, pattern='checkerboard'):
    """Create a test grid with different patterns."""
    if pattern == 'checkerboard':
        grid = np.zeros((size, size), dtype=int)
        grid[::2, ::2] = 1
        grid[1::2, 1::2] = 1
    elif pattern == 'diagonal':
        grid = np.zeros((size, size), dtype=int)
        np.fill_diagonal(grid, 1)
    elif pattern == 'border':
        grid = np.zeros((size, size), dtype=int)
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
    elif pattern == 'random':
        grid = np.random.randint(0, 2, (size, size), dtype=int)
    elif pattern == 'X':
        grid = np.zeros((size, size), dtype=int)
        # Create X pattern
        for i in range(size):
            grid[i, i] = 1  # Main diagonal
            grid[i, size-1-i] = 1  # Anti-diagonal
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    return grid

def plot_selection_result(grid, selection_result, title):
    """Plot the original grid and selection result."""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Create a colormap for the grid (0: white, 1: blue)
    cmap = ListedColormap(['white', 'blue'])
    
    # Plot original grid
    ax1.imshow(grid, cmap=cmap, vmin=0, vmax=1)
    ax1.set_title('Original Grid')
    ax1.grid(True)
    
    # Plot selection result
    if selection_result.ndim == 2:
        # Single 2D mask
        ax2.imshow(selection_result, cmap=cmap, vmin=0, vmax=1)
        ax2.set_title(f'Selection Result\n{title}')
    else:
        # Multiple 2D masks (3D array)
        n_masks = selection_result.shape[0]
        if n_masks > 0:
            # Plot first mask
            ax2.imshow(selection_result[0], cmap=cmap, vmin=0, vmax=1)
            ax2.set_title(f'Selection Result (1 of {n_masks})\n{title}')
        else:
            ax2.imshow(np.zeros_like(grid), cmap=cmap, vmin=0, vmax=1)
            ax2.set_title('Empty Selection Result')
    
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Initialize selectors
    grid_selector = GridSelector()
    
    # Available selection functions
    selection_functions = {
        'all_cells': grid_selector.all_cells,
        'colour': grid_selector.colour,
        'components4': grid_selector.components4,
        'components8': grid_selector.components8,
        'independent_cells_4': grid_selector.independent_cells_4,
        'outer_border4': grid_selector.outer_border4,
        'inner_border4': grid_selector.inner_border4,
        'outer_border8': grid_selector.outer_border8,
        'inner_border8': grid_selector.inner_border8,
        'grid_border': grid_selector.grid_border,
        'contact4_1': grid_selector.contact4_1,
        'contact4_2': grid_selector.contact4_2,
        'contact4_3': grid_selector.contact4_3,
        'contact4_4': grid_selector.contact4_4,
    }
    
    # Print available functions
    print("Available selection functions:")
    for i, func_name in enumerate(selection_functions.keys(), 1):
        print(f"{i}. {func_name}")
    
    # Get user input
    while True:
        try:
            choice = int(input("\nEnter the number of the function to test (0 to exit): "))
            if choice == 0:
                break
            if 1 <= choice <= len(selection_functions):
                func_name = list(selection_functions.keys())[choice-1]
                break
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    if choice == 0:
        return
    
    # Get grid size and pattern
    size = int(input("Enter grid size (default: 5): ") or "5")
    print("\nAvailable patterns:")
    print("1. checkerboard")
    print("2. diagonal")
    print("3. border")
    print("4. random")
    print("5. X")
    
    pattern_choice = int(input("\nEnter pattern number (default: 1): ") or "1")
    patterns = ['checkerboard', 'diagonal', 'border', 'random', 'X']
    pattern = patterns[pattern_choice-1]
    
    # Create test grid
    grid = create_test_grid(size, pattern)
    
    # Get color for color-based functions
    color = int(input("\nEnter color to select (0 or 1, default: 1): ") or "1")
    
    # Apply selection function
    func = selection_functions[func_name]
    try:
        result = func(grid, color)
        plot_selection_result(grid, result, f"{func_name}(color={color})")
        
        # If result is 3D, show all masks
        if result.ndim == 3 and result.shape[0] > 1:
            for i in range(1, result.shape[0]):
                plot_selection_result(grid, result[i], f"{func_name}(color={color}) - Component {i+1}")
    
    except Exception as e:
        print(f"Error applying selection function: {str(e)}")

if __name__ == "__main__":
    main()