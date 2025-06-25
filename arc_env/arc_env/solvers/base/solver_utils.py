from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple # Added Tuple
import numpy as np

# from arc_env.spaces.observation_spaces import ARCStructuredObservation
# from arc_env.dsl.core.base_operations import BaseOperation # If utils help construct op sequences

# This module can contain utility functions that are helpful for implementing
# or evaluating solvers.

def compare_grids(grid1: np.ndarray, grid2: np.ndarray) -> bool:
    """
    Compares two ARC grids for equality. Handles shape differences.
    Returns True if they are identical, False otherwise.
    """
    if grid1.shape != grid2.shape:
        return False
    return np.array_equal(grid1, grid2)

def solution_accuracy(predicted_grid: np.ndarray, target_grid: np.ndarray) -> float:
    """
    Calculates a simple accuracy score based on pixel-wise matching
    between a predicted grid and a target grid.
    If shapes differ, accuracy is 0.
    """
    if not compare_grids(predicted_grid, target_grid): # Handles shape check too
        # If shapes differ, or content differs.
        # If only content differs but shapes match, calculate pixel accuracy.
        if predicted_grid.shape != target_grid.shape:
            return 0.0

        # Calculate pixel-wise accuracy for same-shape differing grids
        if target_grid.size == 0: # Avoid division by zero for empty grids
            return 1.0 if predicted_grid.size == 0 else 0.0 # Both empty is 100% match

        correct_pixels = np.sum(predicted_grid == target_grid)
        total_pixels = target_grid.size
        return float(correct_pixels) / total_pixels
    else: # Grids are identical
        return 1.0


# Example of a utility that might be used by a search-based solver:
# def get_potential_next_states(
#     current_grid: np.ndarray,
#     available_operations: List[BaseOperation], # Or list of decoded action representations
#     selection_mask: Optional[np.ndarray] = None
# ) -> List[Tuple[BaseOperation, np.ndarray]]: # (Operation applied, resulting grid)
#     """
#     Generates potential next grid states by applying a list of operations.
#     This is a core component of many search algorithms.
#     """
#     next_states = []
#     for op in available_operations:
#         try:
#             # Need to handle how selection ops affect the mask for subsequent ops in a sequence.
#             # This simple version assumes each op is applied independently to current_grid/selection_mask.
#             # A real search would manage the evolving selection_mask.
#             if hasattr(op, 'generate_mask') and callable(op.generate_mask) and op.__class__.__name__.lower().startswith("select"):
#                 # If it's a selection op, its "result" is the new mask, not a new grid.
#                 # This utility might not be the right place for this complex logic,
#                 # or it needs to return (op, new_grid, new_mask).
#                 # For now, skip applying selection ops directly for next grid states here.
#                 # Or, assume they don't change the grid passed to this util.
#                 pass # How to handle selection ops needs more thought in context of state generation.
#             else:
#                 next_grid = op.apply(np.copy(current_grid), selection_mask=selection_mask)
#                 next_states.append((op, next_grid))
#         except Exception as e:
#             print(f"SolverUtil: Error applying operation {op.to_string()}: {e}")
#     return next_states


# Placeholder for other utilities:
# - Heuristics for evaluating grid similarity or progress towards solution.
# - Functions to serialize/deserialize solver states or plans.
# - Visualizers for search trees or solution attempts (if not part of rendering).

# Example Usage:
if __name__ == '__main__':
    grid_a = np.array([[1,0],[0,1]])
    grid_b = np.array([[1,0],[0,1]])
    grid_c = np.array([[0,1],[1,0]])
    grid_d = np.array([[1,0,0],[0,1,0],[0,0,1]])

    print(f"grid_a == grid_b: {compare_grids(grid_a, grid_b)}") # True
    print(f"grid_a == grid_c: {compare_grids(grid_a, grid_c)}") # False
    print(f"grid_a == grid_d: {compare_grids(grid_a, grid_d)}") # False (shape diff)

    print(f"\nAccuracy grid_a vs grid_b: {solution_accuracy(grid_a, grid_b)}") # 1.0
    print(f"Accuracy grid_a vs grid_c: {solution_accuracy(grid_a, grid_c)}") # 0.0 (all pixels differ)

    grid_a_mod = np.array([[1,0],[0,2]]) # One pixel different from grid_a
    print(f"Accuracy grid_a vs grid_a_mod: {solution_accuracy(grid_a_mod, grid_a)}") # 3/4 = 0.75

    print(f"Accuracy grid_a vs grid_d: {solution_accuracy(grid_a, grid_d)}") # 0.0 (shape diff)

    empty1 = np.array([[]])
    empty2 = np.array([[]])
    print(f"Accuracy empty1 vs empty2: {solution_accuracy(empty1, empty2)}") # 1.0
    print(f"Accuracy empty1 vs grid_a: {solution_accuracy(empty1, grid_a)}") # 0.0

    # Example with get_potential_next_states (if BaseOperation and some ops were defined here)
    # from arc_env.dsl.core.base_operations import BaseOperation
    # class DummyOp(BaseOperation):
    #     def __init__(self, val_to_add=1): self.val = val_to_add; super().__init__()
    #     def apply(self, grid, selection_mask=None): return grid + self.val
    #     def to_string(self): return f"Add({self.val})"
    #
    # initial_grid = np.array([[0,0],[0,0]])
    # ops_list = [DummyOp(1), DummyOp(5)]
    # potential_states = get_potential_next_states(initial_grid, ops_list)
    # print("\nPotential next states from initial_grid:")
    # for op, next_g in potential_states:
    #     print(f"Applied {op.to_string()}:\n{next_g}")
