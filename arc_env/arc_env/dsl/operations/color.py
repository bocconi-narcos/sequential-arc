from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np

from arc_env.dsl.core.base_operations import BaseOperation
# from arc_env.dsl.utils import color_utils # Assuming color_utils will exist

# Forward reference for type hint in register_color_operations if needed,
# though string literal "OperationRegistry" is also fine.
# if TYPE_CHECKING:
#     from arc_env.dsl.core.operation_registry import OperationRegistry


class ChangeColorOperation(BaseOperation):
    """
    Changes pixels of one color to another color within the selection.
    """
    def _validate_params(self) -> None:
        super()._validate_params()
        required_params = {"from_color": int, "to_color": int}
        for param, p_type in required_params.items():
            if param not in self.params:
                raise ValueError(f"Missing parameter '{param}' for {self.__class__.__name__}")
            if not isinstance(self.params[param], p_type):
                raise ValueError(f"Parameter '{param}' must be of type {p_type.__name__}, "
                                 f"got {type(self.params[param]).__name__}")
        # Add ARC color range validation if needed (e.g., 0-9)
        if not (0 <= self.params["from_color"] <= 9 and 0 <= self.params["to_color"] <= 9):
            raise ValueError("Colors must be integers between 0 and 9.")

    def apply(self, grid: np.ndarray, selection_mask: Optional[np.ndarray] = None) -> np.ndarray:
        new_grid = np.copy(grid)
        from_color = self.params["from_color"]
        to_color = self.params["to_color"]

        if selection_mask is None: # Apply to whole grid
            target_pixels = (new_grid == from_color)
        else: # Apply only within selection
            target_pixels = (new_grid == from_color) & selection_mask

        new_grid[target_pixels] = to_color
        return new_grid

    def to_string(self) -> str:
        return f"ChangeColor(from={self.params['from_color']}, to={self.params['to_color']})"

class FillOperation(BaseOperation):
    """
    Fills the selected area (or whole grid if no selection) with a specified color.
    """
    def _validate_params(self) -> None:
        super()._validate_params()
        if "color" not in self.params or not isinstance(self.params["color"], int):
            raise ValueError(f"Missing or invalid 'color' parameter for {self.__class__.__name__}. Must be int.")
        if not (0 <= self.params["color"] <= 9):
            raise ValueError("Color parameter must be between 0 and 9.")

    def apply(self, grid: np.ndarray, selection_mask: Optional[np.ndarray] = None) -> np.ndarray:
        new_grid = np.copy(grid)
        color_to_apply = self.params["color"]

        if selection_mask is not None:
            new_grid[selection_mask] = color_to_apply
        else:
            # Default behavior: fill the entire grid if no selection_mask
            new_grid[:, :] = color_to_apply
            # Alternative: require selection_mask
            # raise ValueError("FillOperation requires a selection_mask.")
        return new_grid

    def to_string(self) -> str:
        return f"Fill(color={self.params['color']})"


# Add more color operations as needed, e.g.:
# - FloodFillOperation(start_point, target_color, replacement_color)
# - ColorPaletteSwapOperation(palette_map)
# - MostFrequentColorFillOperation() / LeastFrequentColorFillOperation()

def register_color_operations(registry: "OperationRegistry"):
    """Helper function to register all color operations."""
    # Assuming registry is an instance of OperationRegistry
    # from arc_env.dsl.core.operation_registry import OperationRegistry

    registry.register_operation("color", "change_color", ChangeColorOperation)
    registry.register_operation("color", "fill_selected", FillOperation)
    # registry.register_operation("color", "flood_fill", FloodFillOperation)

# Example of how this might be used with the registry:
# if __name__ == '__main__':
#     from arc_env.dsl.core.operation_registry import OperationRegistry
#     op_registry = OperationRegistry()
#     register_color_operations(op_registry)
#     print("Registered color operations:", op_registry.list_available_operations("color"))
#     fill_op_class = op_registry.get_operation_class("color", "fill_selected")
#     if fill_op_class:
#         fill_instance = fill_op_class(color=5)
#         print(fill_instance.to_string())
#         dummy_grid = np.zeros((3,3), dtype=int)
#         mask = np.array([[True, True, False], [True, True, False], [False, False, False]], dtype=bool)
#         print("Original grid:\n", dummy_grid)
#         modified_grid = fill_instance.apply(dummy_grid, selection_mask=mask)
#         print("Grid after fill:\n", modified_grid)

#         change_color_instance = ChangeColorOperation(from_color=0, to_color=3)
#         grid_v2 = np.array([[0,0,1],[0,2,0],[1,0,0]])
#         mask_v2 = np.array([[True, True, False], [True, False, False], [False, True, True]], dtype=bool)
#         print("Original grid v2:\n", grid_v2)
#         modified_grid_v2 = change_color_instance.apply(grid_v2, selection_mask=mask_v2)
#         print("Grid v2 after change_color (selected):\n", modified_grid_v2)
#         modified_grid_v3 = change_color_instance.apply(grid_v2, selection_mask=None) # Whole grid
#         print("Grid v2 after change_color (whole grid):\n", modified_grid_v3)
