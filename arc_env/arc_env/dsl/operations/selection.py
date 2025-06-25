from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple # List, Tuple not directly used in signatures but fine
import numpy as np

from arc_env.dsl.core.base_operations import BaseOperation
# from arc_env.dsl.utils import grid_utils # Assuming grid_utils will exist for complex selections

# if TYPE_CHECKING: # For OperationRegistry type hint
#     from arc_env.dsl.core.operation_registry import OperationRegistry

class SelectByColorOperation(BaseOperation):
    """
    Creates a selection mask for all pixels of a specified color.
    This operation itself doesn't change the grid; it's meant to produce
    a mask that subsequent operations can use.
    However, BaseOperation.apply is expected to return a grid.
    For selection operations, the "returned grid" is typically the original grid,
    and the important output is implicitly the new selection state (which needs
    a mechanism to be passed to the next op, e.g. via an environment state).

    Alternative: Selection operations could return a tuple (grid, new_mask).
    For now, let's assume the environment handles the selection mask state.
    The `apply` method will return the original grid, and it's up to the
    caller (e.g., an interpreter or environment) to capture the mask.
    This might need refinement based on how DSL execution is designed.
    A common pattern is for selection ops to update a "current selection mask"
    in the environment or DSL execution context.
    """
    def _validate_params(self) -> None:
        super()._validate_params()
        if "color" not in self.params or not isinstance(self.params["color"], int):
            raise ValueError(f"Missing or invalid 'color' parameter for {self.__class__.__name__}. Must be int.")
        if not (0 <= self.params["color"] <= 9): # ARC color range
            raise ValueError("Color parameter must be between 0 and 9.")

    def generate_mask(self, grid: np.ndarray, existing_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generates the selection mask.
        Args:
            grid: The input grid.
            existing_mask: An optional existing mask. If provided, the new selection
                           might combine with it (e.g., intersection, union), depending
                           on a mode parameter (not implemented here yet, defaults to new selection).
        Returns:
            A boolean numpy array representing the new selection.
        """
        # By default, this creates a new selection.
        # Could add modes: "new", "add", "subtract", "intersect" with existing_mask.
        return grid == self.params["color"]

    def apply(self, grid: np.ndarray, selection_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Applies the selection. For selection operations, this means determining
        the new selection mask. The grid itself is typically returned unchanged.
        The new mask should be handled by the DSL execution engine.

        This method is a bit problematic for pure selection ops if the BaseOperation
        interface is strictly (grid_in) -> (grid_out). The "effect" is the new mask.
        For now, we return the grid unmodified and expect the caller to use generate_mask.
        """
        # The "application" of a selection operation is to define a new selection mask.
        # The grid itself is not modified by this operation.
        # The environment or DSL interpreter is responsible for taking the output of
        # `self.generate_mask(grid, selection_mask)` and using it as the
        # `selection_mask` for subsequent operations.
        return np.copy(grid) # Return unchanged grid

    def to_string(self) -> str:
        return f"SelectByColor(color={self.params['color']})"

class SelectAllOperation(BaseOperation):
    """Selects all pixels in the grid."""

    def generate_mask(self, grid: np.ndarray, existing_mask: Optional[np.ndarray] = None) -> np.ndarray:
        return np.ones_like(grid, dtype=bool)

    def apply(self, grid: np.ndarray, selection_mask: Optional[np.ndarray] = None) -> np.ndarray:
        return np.copy(grid)

    def to_string(self) -> str:
        return "SelectAll()"

class ClearSelectionOperation(BaseOperation):
    """Clears any existing selection (results in an all-False mask)."""

    def generate_mask(self, grid: np.ndarray, existing_mask: Optional[np.ndarray] = None) -> np.ndarray:
        return np.zeros_like(grid, dtype=bool)

    def apply(self, grid: np.ndarray, selection_mask: Optional[np.ndarray] = None) -> np.ndarray:
        return np.copy(grid)

    def to_string(self) -> str:
        return "ClearSelection()"

class InvertSelectionOperation(BaseOperation):
    """Inverts the current selection mask."""

    def generate_mask(self, grid: np.ndarray, existing_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if existing_mask is None:
            # If no prior selection, inverting it means selecting everything.
            # This behavior might need to be defined more clearly.
            # Or, it could require an existing_mask.
            # print("Warning: Inverting selection when no prior selection exists. Selecting all.")
            return np.ones_like(grid, dtype=bool)
        return ~existing_mask

    def apply(self, grid: np.ndarray, selection_mask: Optional[np.ndarray] = None) -> np.ndarray:
        return np.copy(grid)

    def to_string(self) -> str:
        return "InvertSelection()"


# TODO: Add more selection operations:
# - SelectByPatternOperation(pattern_grid)
# - SelectByObjectBoundaryOperation()
# - SelectLargestComponentOperation(color_filter=None)
# - SelectByCoordinatesOperation(points: List[Tuple[int,int]]) / SelectByAreaOperation(rect: Tuple[int,int,int,int])
# - CombineSelectionOperation(mask1, mask2, mode="union"|"intersection"|"difference")
#   (This one is tricky as it implies multiple active masks or a history)


def register_selection_operations(registry: "OperationRegistry"):
    """Helper function to register all selection operations."""
    # from arc_env.dsl.core.operation_registry import OperationRegistry (if type hint needed)

    registry.register_operation("selection", "select_color", SelectByColorOperation)
    registry.register_operation("selection", "select_all", SelectAllOperation)
    registry.register_operation("selection", "clear_selection", ClearSelectionOperation)
    registry.register_operation("selection", "invert_selection", InvertSelectionOperation)
    # ... register others ...

# Example Usage:
# if __name__ == '__main__':
#     from arc_env.dsl.core.operation_registry import OperationRegistry
#     op_registry = OperationRegistry()
#     register_selection_operations(op_registry)
#     print("Registered selection operations:", op_registry.list_available_operations("selection"))

#     grid_example = np.array([
#         [1, 1, 0],
#         [0, 2, 1],
#         [2, 2, 0]
#     ], dtype=int)

#     select_color_op = op_registry.get_operation_class("selection", "select_color")(color=1)
#     if select_color_op: # Check if class was retrieved
#         print(f"\n{select_color_op.to_string()}")
#         mask = select_color_op.generate_mask(grid_example)
#         print("Mask for color 1:\n", mask)
#         # Grid remains unchanged by the selection op itself
#         # modified_grid = select_color_op.apply(grid_example)
#         # print("Grid after select_color_op.apply (should be unchanged):\n", modified_grid)

#     select_all_op = op_registry.get_operation_class("selection", "select_all")()
#     if select_all_op:
#         print(f"\n{select_all_op.to_string()}")
#         mask_all = select_all_op.generate_mask(grid_example)
#         print("Mask for select_all:\n", mask_all)

#     invert_op = op_registry.get_operation_class("selection", "invert_selection")()
#     if invert_op and 'mask' in locals(): # Use previously generated mask
#         print(f"\n{invert_op.to_string()}")
#         inverted_mask = invert_op.generate_mask(grid_example, existing_mask=mask)
#         print("Inverted mask (of color 1 selection):\n", inverted_mask)
