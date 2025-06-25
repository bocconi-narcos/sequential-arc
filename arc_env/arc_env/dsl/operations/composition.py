from __future__ import annotations

from typing import Optional, Dict, Any, List # List not used in signatures but fine
import numpy as np

from arc_env.dsl.core.base_operations import BaseOperation
# from arc_env.dsl.utils import grid_utils # Assuming grid_utils will exist

# if TYPE_CHECKING: # For OperationRegistry type hint
#     from arc_env.dsl.core.operation_registry import OperationRegistry


# Note: Composition operations are often the most complex as they might involve
# multiple grids, a clipboard/buffer, or advanced object detection and alignment.
# The examples here will be simplified.

class CopyPasteOperation(BaseOperation):
    """
    "Copies" the selected area and "pastes" it, possibly to a new location
    or with transformations. This is a complex operation that implies
    a clipboard mechanism and a way to specify the paste target.

    Simplified version: Copies the selection and pastes it back onto the
    grid at specified target coordinates (top-left of paste).
    Overlapping parts are overwritten.
    The "copy" part is implicit in taking the selection.
    """
    def _validate_params(self) -> None:
        super()._validate_params()
        # Target coordinates for the top-left of the pasted content
        if "target_row" not in self.params or not isinstance(self.params["target_row"], int):
            raise ValueError("Missing or invalid 'target_row' parameter (int).")
        if "target_col" not in self.params or not isinstance(self.params["target_col"], int):
            raise ValueError("Missing or invalid 'target_col' parameter (int).")
        # Optional: "ignore_color" for transparency when pasting (int or None)
        self.params.setdefault("ignore_color", None)
        if self.params["ignore_color"] is not None and not isinstance(self.params["ignore_color"], int):
            raise ValueError("'ignore_color' must be an int if provided.")


    def apply(self, grid: np.ndarray, selection_mask: Optional[np.ndarray] = None) -> np.ndarray:
        new_grid = np.copy(grid)

        if selection_mask is None or not np.any(selection_mask):
            # print("Warning: CopyPasteOperation called with no selection. No action taken.")
            return new_grid # Nothing to copy

        rows, cols = np.where(selection_mask)
        r_min, r_max = np.min(rows), np.max(rows)
        c_min, c_max = np.min(cols), np.max(cols)

        # This is the content to be "copied"
        copied_subgrid = grid[r_min:r_max+1, c_min:c_max+1]
        # The selection relative to the subgrid itself (for ignoring parts outside selection)
        relative_selection_mask = selection_mask[r_min:r_max+1, c_min:c_max+1]

        target_r, target_c = self.params["target_row"], self.params["target_col"]
        ignore_color = self.params["ignore_color"]

        # Paste logic
        subgrid_h, subgrid_w = copied_subgrid.shape
        grid_h, grid_w = new_grid.shape

        for r_offset in range(subgrid_h):
            for c_offset in range(subgrid_w):
                # Only paste if the pixel was part of the original selection within the subgrid
                if not relative_selection_mask[r_offset, c_offset]:
                    continue

                pixel_to_paste = copied_subgrid[r_offset, c_offset]

                # Skip pasting if it's the "transparent" ignore_color
                if ignore_color is not None and pixel_to_paste == ignore_color:
                    continue

                # Determine actual coordinates on the main grid
                dest_r, dest_c = target_r + r_offset, target_c + c_offset

                # Check bounds
                if 0 <= dest_r < grid_h and 0 <= dest_c < grid_w:
                    new_grid[dest_r, dest_c] = pixel_to_paste

        return new_grid

    def to_string(self) -> str:
        s = f"CopyPaste(target_row={self.params['target_row']}, target_col={self.params['target_col']}"
        if self.params['ignore_color'] is not None:
            s += f", ignore_color={self.params['ignore_color']}"
        s += ")"
        return s


class OverlayOperation(BaseOperation):
    """
    Overlays a "pattern" grid onto the main grid, potentially with transparency.
    This implies the pattern is available (e.g., from parameters or a buffer).

    Simplified: The "pattern" is the current selection, and it's overlaid
    onto the grid, respecting a transparent color. This is very similar to
    CopyPaste with ignore_color if target_row/col are the selection's origin.
    A more distinct Overlay would take another grid as input.

    Let's make this one "stamp" the current selection, treating a specific color
    within the selection as transparent.
    """
    def _validate_params(self) -> None:
        super()._validate_params()
        if "transparent_color" not in self.params or not isinstance(self.params["transparent_color"], int):
            raise ValueError("Missing or invalid 'transparent_color' parameter (int).")
        # Optional: target coordinates, if not specified, overlays in place.
        self.params.setdefault("target_row", None) # If None, use selection's original position
        self.params.setdefault("target_col", None)

        if (self.params["target_row"] is not None and not isinstance(self.params["target_row"], int)) or \
           (self.params["target_col"] is not None and not isinstance(self.params["target_col"], int)):
            raise ValueError("target_row/col must be ints if provided for Overlay.")


    def apply(self, grid: np.ndarray, selection_mask: Optional[np.ndarray] = None) -> np.ndarray:
        new_grid = np.copy(grid)

        if selection_mask is None or not np.any(selection_mask):
            return new_grid

        rows, cols = np.where(selection_mask)
        r_min, r_max = np.min(rows), np.max(rows)
        c_min, c_max = np.min(cols), np.max(cols)

        stamp_subgrid = grid[r_min:r_max+1, c_min:c_max+1]
        relative_selection_mask = selection_mask[r_min:r_max+1, c_min:c_max+1]

        transparent_color = self.params["transparent_color"]

        # Determine target paste position
        paste_r_start = self.params["target_row"] if self.params["target_row"] is not None else r_min
        paste_c_start = self.params["target_col"] if self.params["target_col"] is not None else c_min

        subgrid_h, subgrid_w = stamp_subgrid.shape
        grid_h, grid_w = new_grid.shape

        for r_offset in range(subgrid_h):
            for c_offset in range(subgrid_w):
                if not relative_selection_mask[r_offset, c_offset]: # Only consider selected parts of the stamp
                    continue

                pixel_to_stamp = stamp_subgrid[r_offset, c_offset]

                if pixel_to_stamp == transparent_color:
                    continue # Skip transparent pixels

                dest_r, dest_c = paste_r_start + r_offset, paste_c_start + c_offset

                if 0 <= dest_r < grid_h and 0 <= dest_c < grid_w:
                    new_grid[dest_r, dest_c] = pixel_to_stamp

        return new_grid

    def to_string(self) -> str:
        s = f"Overlay(transparent_color={self.params['transparent_color']}"
        if self.params['target_row'] is not None:
            s += f", target_row={self.params['target_row']}, target_col={self.params['target_col']}"
        s += ")"
        return s

# TODO: Add more composition operations:
# - CombineObjectsOperation (e.g., union of two selected objects, handling overlaps)
# - StackOperation (if multiple objects are selected, stack them along an axis)
# - TileOperation (tile the selected pattern across the grid or a larger selection)
# - InsertObjectOperation (takes a full object representation and places it)

def register_composition_operations(registry: "OperationRegistry"):
    """Helper function to register all composition operations."""
    registry.register_operation("composition", "copy_paste", CopyPasteOperation)
    registry.register_operation("composition", "overlay_selection", OverlayOperation)
    # ... register others ...

# Example:
# if __name__ == '__main__':
#     from arc_env.dsl.core.operation_registry import OperationRegistry
#     op_registry = OperationRegistry()
#     register_composition_operations(op_registry)
#     print("Registered composition operations:", op_registry.list_available_operations("composition"))

#     grid_ex = np.array([
#         [1, 1, 0, 0],
#         [1, 1, 0, 0],
#         [0, 0, 2, 2],
#         [0, 0, 2, 2]
#     ])
#     # Select top-left 2x2 object (the 1s)
#     mask_ex = np.zeros_like(grid_ex, dtype=bool)
#     mask_ex[0:2, 0:2] = True

#     copypaste_op = op_registry.get_operation_class("composition", "copy_paste")(target_row=2, target_col=2, ignore_color=0)
#     if copypaste_op:
#         print(f"\n{copypaste_op.to_string()}")
#         pasted_grid = copypaste_op.apply(grid_ex.copy(), selection_mask=mask_ex)
#         print("Grid after CopyPaste (1s to bottom-right, 0 is transparent if part of selection):\n", pasted_grid)
#         # Expected:
#         # [[1, 1, 0, 0],
#         #  [1, 1, 0, 0],
#         #  [0, 0, 1, 1],
#         #  [0, 0, 1, 1]]

#     grid_ex_overlay = np.array([
#         [5, 0, 0], # Object is [[5,0],[0,5]], 0 is transparent
#         [0, 5, 0],
#         [8, 8, 8]
#     ])
#     mask_overlay = np.zeros_like(grid_ex_overlay, dtype=bool)
#     mask_overlay[0:2,0:2] = True # Select the 2x2 area containing the 5s and 0s

#     overlay_op = op_registry.get_operation_class("composition", "overlay_selection")(transparent_color=0)
#     if overlay_op:
#         print(f"\n{overlay_op.to_string()}")
#         # Overlay the selection (the 5s) in place, 0s are transparent
#         overlaid_grid = overlay_op.apply(grid_ex_overlay.copy(), selection_mask=mask_overlay)
#         print("Grid after Overlay (in place, 0 is transparent):\n", overlaid_grid)
#         # Expected: (assuming original selection was the 2x2 top left)
#         # [[5, 0, 0], # If background was 0, then 5, (original 0), 0
#         #  [0, 5, 0], # (original 0), 5, (original 0)
#         #  [8, 8, 8]]
#         # This means if a 0 from selection is overlayed on 8, 8 remains. If 5 on 8, 8 becomes 5.
#         # Original grid to show effect:
#         # grid_ex_overlay_target = np.array([
#         #    [8, 8, 8],
#         #    [8, 8, 8],
#         #    [8, 8, 8]
#         # ])
#         # overlay_op_target = op_registry.get_operation_class("composition", "overlay_selection")(
#         #     transparent_color=0, target_row=0, target_col=0
#         # )
#         # overlaid_grid_target = overlay_op_target.apply(grid_ex_overlay_target.copy(), selection_mask=mask_overlay)
#         # print("Grid after Overlay (on 8s, 0 is transparent):\n", overlaid_grid_target)
#         # Expected:
#         # [[5, 8, 8],
#         #  [8, 5, 8],
#         #  [8, 8, 8]]
