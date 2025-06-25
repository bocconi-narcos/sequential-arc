from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional # Tuple not used directly here, but good for general typing
import numpy as np

class BaseOperation(ABC):
    """
    Abstract base class for all Domain-Specific Language (DSL) operations.

    Each operation represents a transformation or query that can be applied
    to an ARC grid or a selection within it.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the operation.
        Kwargs can be used to pass parameters specific to the operation,
        which should be validated.
        """
        self.params = kwargs
        self._validate_params()

    def _validate_params(self) -> None:
        """
        Validates the parameters passed during initialization.
        Subclasses should override this to define their expected parameters
        and validation logic.
        Example:
            expected_params = {"color": int, "target_area": str}
            for p_name, p_type in expected_params.items():
                if p_name not in self.params:
                    raise ValueError(f"Missing parameter '{p_name}' for {self.__class__.__name__}")
                if not isinstance(self.params[p_name], p_type):
                    raise ValueError(f"Parameter '{p_name}' must be of type {p_type.__name__}, "
                                     f"got {type(self.params[p_name]).__name__}")
        """
        pass # Default implementation: no parameters expected or validated.

    @abstractmethod
    def apply(self, grid: np.ndarray, selection_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Applies the operation to the given grid.

        Args:
            grid: The input ARC grid (2D numpy array of integers representing colors).
            selection_mask: An optional boolean numpy array of the same shape as `grid`.
                            If provided, the operation should primarily (or exclusively)
                            affect the cells where the mask is True. If None, the operation
                            might apply to the whole grid or a default region, depending
                            on its nature.

        Returns:
            A new grid representing the state after the operation is applied.
            Operations should strive to be non-mutating to the input grid by default,
            returning a copy.
        """
        pass

    def __call__(self, grid: np.ndarray, selection_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Makes the operation instance callable, delegating to the apply method."""
        return self.apply(grid, selection_mask)

    @abstractmethod
    def to_string(self) -> str:
        """
        Returns a concise string representation of the operation, including its
        key parameters. Useful for logging, debugging, or representing a sequence
        of operations.

        Example: "Color(color=1, area='selected_pixels')"
        """
        pass

    def __repr__(self) -> str:
        return self.to_string()

    def get_params(self) -> Dict[str, Any]:
        """Returns the parameters of the operation."""
        return self.params.copy()

    # Optional: Add methods for checking applicability, cost, or other metadata
    # def is_applicable(self, grid: np.ndarray, selection_mask: Optional[np.ndarray] = None) -> bool:
    #     """Checks if the operation can be meaningfully applied to the current state."""
    #     return True # Default: always applicable

    # def get_cost(self) -> float:
    #     """Returns an estimated cost or complexity of applying this operation."""
    #     return 1.0 # Default cost


# Example of a concrete operation (for illustration, not part of this file)
# class FillColor(BaseOperation):
#     def _validate_params(self) -> None:
#         if "color" not in self.params or not isinstance(self.params["color"], int):
#             raise ValueError("FillColor operation requires an integer 'color' parameter.")
#         if not (0 <= self.params["color"] <= 9): # Assuming ARC colors 0-9
#             raise ValueError("Color parameter must be between 0 and 9.")

#     def apply(self, grid: np.ndarray, selection_mask: Optional[np.ndarray] = None) -> np.ndarray:
#         new_grid = np.copy(grid)
#         color_to_apply = self.params["color"]
#         if selection_mask is not None:
#             new_grid[selection_mask] = color_to_apply
#         else:
#             # If no selection, fill the entire grid (or define other default behavior)
#             new_grid[:, :] = color_to_apply
#         return new_grid

#     def to_string(self) -> str:
#         return f"FillColor(color={self.params['color']})"
