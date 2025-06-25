from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np

from arc_env.dsl.core.base_operations import BaseOperation
# from arc_env.dsl.utils import grid_utils # Assuming grid_utils will exist for complex transformations

# if TYPE_CHECKING: # For OperationRegistry type hint
#     from arc_env.dsl.core.operation_registry import OperationRegistry

class RotateOperation(BaseOperation):
    """
    Rotates the selected area (or whole grid) by a specified angle.
    """
    def _validate_params(self) -> None:
        super()._validate_params()
        if "angle" not in self.params or not isinstance(self.params["angle"], int):
            raise ValueError(f"Missing or invalid 'angle' parameter for {self.__class__.__name__}. Must be int.")
        if self.params["angle"] not in [90, 180, 270, -90, -180, -270]: # Common rotations
            # Could also support arbitrary angles if image processing libraries are used,
            # but for ARC, discrete rotations are typical.
            raise ValueError("Angle must be one of 90, 180, 270 (or negative equivalents).")

    def apply(self, grid: np.ndarray, selection_mask: Optional[np.ndarray] = None) -> np.ndarray:
        new_grid = np.copy(grid)
        angle = self.params["angle"]

        # Determine k for np.rot90: 1 for 90 deg clockwise, 2 for 180, 3 for 270 clockwise (-90)
        # np.rot90 rotates counter-clockwise.
        if angle == 90: k = 3 # 3 counter-clockwise rotations = 1 clockwise
        elif angle == 180: k = 2
        elif angle == 270: k = 1
        elif angle == -90: k = 1
        elif angle == -180: k = 2
        elif angle == -270: k = 3
        else: k = 0 # Should not happen due to validation

        if k == 0: return new_grid # No rotation

        if selection_mask is None:
            # Rotate the whole grid
            rotated_part = np.rot90(new_grid, k=k)
            # This can change the shape of the grid. ARC usually expects fixed canvas.
            # For whole grid rotations, it's often assumed the grid is square
            # or padding/cropping occurs. Let's assume square or fitting rotation for now.
            # If not square, np.rot90 handles it by changing dimensions.
            # This needs careful consideration for ARC's fixed canvas.
            # A common simplification is to only allow rotation of square selections or objects.
            if new_grid.shape != rotated_part.shape:
                # This is a problem for fixed-canvas ARC.
                # Solutions:
                # 1. Only allow on square grids/selections.
                # 2. Pad/crop to fit (complex, might lose info or add undesired background).
                # 3. Define a bounding box of the "content" and rotate that, then place back.
                print(f"Warning: Whole grid rotation changed shape from {new_grid.shape} to {rotated_part.shape}. "
                      "This might not be intended for fixed-canvas environments.")
                # For now, we'll return the rotated grid as is. This needs a better strategy.
                return rotated_part # This is problematic if shape changes.
            return rotated_part

        # If there is a selection_mask, the behavior is more complex:
        # 1. Extract the subgrid defined by the bounding box of the selection.
        # 2. Rotate this subgrid.
        # 3. Place the rotated subgrid back into the main grid.
        # This requires careful handling of coordinates and potential overwriting.

        # Simplified approach: If selection is not the whole grid, this op might be hard to define
        # robustly without a clear "object" concept from the selection.
        # For now, let's assume if a selection_mask is provided, we attempt to rotate
        # the bounding box of that selection. This is a common interpretation.

        # Find bounding box of selection_mask
        rows, cols = np.where(selection_mask)
        if not rows.size: return new_grid # Empty selection, no change

        r_min, r_max = np.min(rows), np.max(rows)
        c_min, c_max = np.min(cols), np.max(cols)

        subgrid = new_grid[r_min:r_max+1, c_min:c_max+1]
        rotated_subgrid = np.rot90(subgrid, k=k)

        # Check if the rotated subgrid fits back into its original bounding box space.
        # This is only true if the subgrid was square and rotation is by 90/180/270.
        # If not square, the rotated shape will differ.
        if rotated_subgrid.shape == subgrid.shape:
            new_grid[r_min:r_max+1, c_min:c_max+1] = rotated_subgrid
        else:
            # More complex: The rotated part has a new shape. Where does it go?
            # Option A: Fail the operation.
            # Option B: Try to place it centered at the original center (might go out of bounds or overwrite).
            # Option C: The operation is only valid if the selection is square.
            # For now, let's print a warning and not apply if shapes don't match.
            # This is a strong indicator that `grid_utils` for object manipulation are needed.
            print(f"Warning: Rotating selection of shape {subgrid.shape} resulted in new shape {rotated_subgrid.shape}. "
                  "Cannot place back into original bounding box. Grid remains unchanged for this part.")
            # To make it "safer", perhaps it should only apply if it's a square selection
            if subgrid.shape[0] != subgrid.shape[1]:
                print("Rotation of non-square selection is ambiguous with fixed bounding box. No change applied to selection.")
                return new_grid # No change if not square, to avoid shape errors.
            # If it was square, but rotation changed shape (should not happen with rot90), it's an issue.
            # This implies the subgrid itself must be square for rot90 to preserve shape of bounding box.

            # A simple "place back" if it was square.
            if subgrid.shape[0] == subgrid.shape[1]: # If it was square
                 new_grid[r_min:r_max+1, c_min:c_max+1] = rotated_subgrid
            # This whole block for selection needs a more robust definition or helper utils.

        return new_grid

    def to_string(self) -> str:
        return f"Rotate(angle={self.params['angle']})"


class FlipOperation(BaseOperation):
    """
    Flips the selected area (or whole grid) horizontally or vertically.
    """
    def _validate_params(self) -> None:
        super()._validate_params()
        if "axis" not in self.params or self.params["axis"] not in ["horizontal", "vertical", 0, 1]:
            raise ValueError(f"Missing or invalid 'axis' parameter for {self.__class__.__name__}. "
                             "Must be 'horizontal' (or 1) or 'vertical' (or 0).")
        # Convert named axis to numpy axis convention if needed
        if self.params["axis"] == "vertical": self.params["axis"] = 0 # Flip along rows (top-bottom)
        elif self.params["axis"] == "horizontal": self.params["axis"] = 1 # Flip along columns (left-right)


    def apply(self, grid: np.ndarray, selection_mask: Optional[np.ndarray] = None) -> np.ndarray:
        new_grid = np.copy(grid)
        axis = self.params["axis"]

        if selection_mask is None:
            # Flip the whole grid
            if axis == 0: # Vertical flip
                new_grid = np.flip(new_grid, axis=0)
            elif axis == 1: # Horizontal flip
                new_grid = np.flip(new_grid, axis=1)
            return new_grid

        # Flip only the selected area (bounding box)
        rows, cols = np.where(selection_mask)
        if not rows.size: return new_grid # Empty selection

        r_min, r_max = np.min(rows), np.max(rows)
        c_min, c_max = np.min(cols), np.max(cols)

        subgrid_selection = new_grid[r_min:r_max+1, c_min:c_max+1]

        if axis == 0: # Vertical flip
            flipped_subgrid = np.flip(subgrid_selection, axis=0)
        elif axis == 1: # Horizontal flip
            flipped_subgrid = np.flip(subgrid_selection, axis=1)
        else: # Should not happen
            return new_grid

        new_grid[r_min:r_max+1, c_min:c_max+1] = flipped_subgrid
        return new_grid

    def to_string(self) -> str:
        axis_name = self.params['axis']
        if axis_name == 0: axis_name = "vertical"
        elif axis_name == 1: axis_name = "horizontal"
        return f"Flip(axis='{axis_name}')"

# TODO: Add more transformation operations:
# - MoveOperation(dx, dy, wrap=False) / ShiftOperation
# - ScaleOperation(factor, interpolation='nearest') (complex for discrete grids)
# - CropOperation(to_selection_bbox=True, padding=0) / ExtendOperation
# - ShearOperation (complex)
# - ResampleGridOperation(new_shape)

def register_transformation_operations(registry: "OperationRegistry"):
    """Helper function to register all transformation operations."""
    registry.register_operation("transform", "rotate", RotateOperation)
    registry.register_operation("transform", "flip", FlipOperation)
    # ... register others ...

# Example:
# if __name__ == '__main__':
#     from arc_env.dsl.core.operation_registry import OperationRegistry
#     op_registry = OperationRegistry()
#     register_transformation_operations(op_registry)
#     print("Registered transformation operations:", op_registry.list_available_operations("transform"))

#     grid_ex = np.array([
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]
#     ])
#     mask_ex = np.zeros_like(grid_ex, dtype=bool)
#     mask_ex[0:2, 0:2] = True # Select top-left 2x2 square
#     # [[True, True, False],
#     #  [True, True, False],
#     #  [False,False,False]]

#     rotate_op = op_registry.get_operation_class("transform", "rotate")(angle=90)
#     if rotate_op:
#         print(f"\n{rotate_op.to_string()}")
#         # Rotate whole grid (can change shape if not square, or be problematic)
#         # For a 3x3 grid, rotating 90 deg keeps shape 3x3.
#         # rotated_grid_all = rotate_op.apply(grid_ex.copy())
#         # print("Rotated whole grid (90 deg clockwise):\n", rotated_grid_all)

#         # Rotate selection
#         rotated_grid_sel = rotate_op.apply(grid_ex.copy(), selection_mask=mask_ex)
#         print("Grid with 2x2 selection rotated (90 deg clockwise):\n", rotated_grid_sel)
#         # Expected: [[4,1,3],[5,2,6],[7,8,9]] if top-left 2x2 [[1,2],[4,5]] rotates to [[4,1],[5,2]]

#     flip_op = op_registry.get_operation_class("transform", "flip")(axis="horizontal")
#     if flip_op:
#         print(f"\n{flip_op.to_string()}")
#         # Flipped whole grid
#         # flipped_grid_all = flip_op.apply(grid_ex.copy())
#         # print("Flipped whole grid (horizontal):\n", flipped_grid_all)

#         # Flipped selection
#         flipped_grid_sel = flip_op.apply(grid_ex.copy(), selection_mask=mask_ex)
#         print("Grid with 2x2 selection flipped (horizontal):\n", flipped_grid_sel)
#         # Expected: [[2,1,3],[5,4,6],[7,8,9]] if top-left 2x2 [[1,2],[4,5]] flips to [[2,1],[5,4]]
