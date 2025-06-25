from __future__ import annotations

import numpy as np
from typing import Any, List, Dict, Optional

from .base import BaseDataProcessor
from arc_env.data.loaders.base import ARCTaskData # Protocol for task data structure
from arc_env.exceptions import DataProcessingError, ConfigurationError # Added ConfigurationError for config validation

class ARCTaskValidator(BaseDataProcessor[ARCTaskData, bool]):
    """
    A data processor that validates the structure and content of ARCTaskData.
    It checks for common issues like incorrect grid dimensions, invalid color values,
    missing pairs, etc.
    """

    def __init__(
        self,
        min_colors: int = 0,
        max_colors: int = 9,
        expected_grid_ndim: int = 2,
        require_train_pairs: bool = True,
        require_test_pairs: bool = True,
        allow_empty_grids: bool = False, # ARC sometimes has empty grids (e.g. shape (0,N) or (N,0))
        **kwargs: Any
    ):
        """
        Args:
            min_colors: Minimum valid color value (inclusive).
            max_colors: Maximum valid color value (inclusive).
            expected_grid_ndim: Expected number of dimensions for grids (typically 2 for ARC).
            require_train_pairs: If True, task must have at least one training pair.
            require_test_pairs: If True, task must have at least one test pair.
            allow_empty_grids: If False, grids with a zero dimension (e.g. shape (0,5)) are invalid.
        """
        super().__init__(**kwargs) # Pass any other configs to base
        self.min_colors = min_colors
        self.max_colors = max_colors
        self.expected_grid_ndim = expected_grid_ndim
        self.require_train_pairs = require_train_pairs
        self.require_test_pairs = require_test_pairs
        self.allow_empty_grids = allow_empty_grids
        self.validation_errors: List[str] = []

    def _validate_config(self) -> None:
        super()._validate_config()
        if self.min_colors > self.max_colors:
            raise ConfigurationError("min_colors cannot be greater than max_colors.")
        if self.expected_grid_ndim < 1:
            raise ConfigurationError("expected_grid_ndim must be at least 1.")

    def _log_error(self, message: str) -> None:
        self.validation_errors.append(message)

    def _validate_grid(self, grid: np.ndarray, grid_name: str, task_id: Optional[str]) -> bool:
        """Validates a single grid."""
        is_valid = True
        tid_prefix = f"Task '{task_id or 'Unknown'}' - {grid_name}:"

        if not isinstance(grid, np.ndarray):
            self._log_error(f"{tid_prefix} is not a numpy array (type: {type(grid).__name__}).")
            return False # Cannot proceed with other checks

        if grid.ndim != self.expected_grid_ndim:
            self._log_error(f"{tid_prefix} has incorrect dimensions (expected {self.expected_grid_ndim}, got {grid.ndim}).")
            is_valid = False

        if not self.allow_empty_grids and (grid.shape[0] == 0 or grid.shape[1] == 0 if grid.ndim == 2 else grid.size == 0):
            self._log_error(f"{tid_prefix} is empty (shape: {grid.shape}) and allow_empty_grids is False.")
            is_valid = False
        elif self.allow_empty_grids and (grid.shape[0] == 0 or grid.shape[1] == 0 if grid.ndim == 2 else grid.size == 0):
            pass # Empty grid is allowed, skip color checks.
        elif grid.size > 0: # Only check colors if grid is not empty
            if not np.all((grid >= self.min_colors) & (grid <= self.max_colors)):
                # Find specific out-of-range colors for better error message
                unique_vals = np.unique(grid)
                out_of_range = unique_vals[(unique_vals < self.min_colors) | (unique_vals > self.max_colors)]
                self._log_error(f"{tid_prefix} contains invalid color values. Expected [{self.min_colors}-{self.max_colors}]. Found: {out_of_range.tolist()}.")
                is_valid = False

        return is_valid

    def process(self, data: ARCTaskData) -> bool:
        """
        Validates the ARCTaskData.

        Args:
            data: The ARCTaskData object to validate.

        Returns:
            True if the data is valid, False otherwise.
            Detailed errors are stored in `self.validation_errors`.
        """
        self.validation_errors = [] # Reset errors for this run
        task_id = data.task_id if hasattr(data, 'task_id') else "N/A"

        if not hasattr(data, 'train') or not isinstance(data.train, list): # type: ignore
            self._log_error(f"Task '{task_id}': 'train' field missing or not a list.")
            return False
        if not hasattr(data, 'test') or not isinstance(data.test, list): # type: ignore
            self._log_error(f"Task '{task_id}': 'test' field missing or not a list.")
            return False

        if self.require_train_pairs and not data.train: # type: ignore
            self._log_error(f"Task '{task_id}': No training pairs provided, but require_train_pairs is True.")
        if self.require_test_pairs and not data.test: # type: ignore
            self._log_error(f"Task '{task_id}': No test pairs provided, but require_test_pairs is True.")

        # If errors found already (e.g. missing train/test lists), no point continuing grid checks
        if self.validation_errors:
            return False

        overall_valid = True
        for pair_type, pairs in [("Train", data.train), ("Test", data.test)]: # type: ignore
            for i, pair in enumerate(pairs):
                if not isinstance(pair, dict):
                    self._log_error(f"Task '{task_id}': {pair_type} pair {i} is not a dictionary.")
                    overall_valid = False; continue

                if "input" not in pair:
                    self._log_error(f"Task '{task_id}': {pair_type} pair {i} is missing 'input' grid.")
                    overall_valid = False
                elif not self._validate_grid(pair["input"], f"{pair_type} pair {i} input", task_id):
                    overall_valid = False

                if "output" not in pair:
                    self._log_error(f"Task '{task_id}': {pair_type} pair {i} is missing 'output' grid.")
                    overall_valid = False
                elif not self._validate_grid(pair["output"], f"{pair_type} pair {i} output", task_id):
                    overall_valid = False

        if not overall_valid and not self.validation_errors: # Should not happen if logic is correct
             self._log_error(f"Task '{task_id}': Unknown validation failure occurred.")

        return not bool(self.validation_errors) # True if no errors logged

    def get_errors(self) -> List[str]:
        """Returns the list of validation errors from the last `process` call."""
        return self.validation_errors

# Example Usage:
# if __name__ == "__main__":
#     from dataclasses import dataclass

#     @dataclass
#     class DummyTask: # Implements ARCTaskData protocol
#         train: List[Dict[str, np.ndarray]]
#         test: List[Dict[str, np.ndarray]]
#         task_id: Optional[str] = None

#     validator = ARCTaskValidator(allow_empty_grids=True)

#     # Valid task
#     valid_task = DummyTask(
#         task_id="valid_01",
#         train=[{"input": np.array([[1,2],[3,0]]), "output": np.array([[0,1]])}],
#         test=[{"input": np.array([[5]]), "output": np.array([[4,4,4]])}]
#     )
#     is_valid = validator.process(valid_task)
#     print(f"Validation for 'valid_01': {is_valid}")
#     if not is_valid: print("Errors:", validator.get_errors())

#     # Invalid task: wrong color
#     invalid_color_task = DummyTask(
#         task_id="invalid_color_02",
#         train=[{"input": np.array([[1, 10]]), "output": np.array([[0]])}], # 10 is out of range 0-9
#         test=[{"input": np.array([[1]]), "output": np.array([[-1]])}]   # -1 is out of range
#     )
#     is_valid = validator.process(invalid_color_task)
#     print(f"\nValidation for 'invalid_color_02': {is_valid}")
#     if not is_valid: print("Errors:", "\n - ".join(validator.get_errors()))

#     # Invalid task: wrong dimensions
#     invalid_dim_task = DummyTask(
#         task_id="invalid_dim_03",
#         train=[{"input": np.array([1,2,3]), "output": np.array([[0]])}], # 1D input
#         test=[{"input": np.array([[[1]]]), "output": np.array([[0]])}]  # 3D input
#     )
#     is_valid = validator.process(invalid_dim_task)
#     print(f"\nValidation for 'invalid_dim_03': {is_valid}")
#     if not is_valid: print("Errors:", "\n - ".join(validator.get_errors()))

#     # Invalid task: missing pairs
#     missing_pairs_task = DummyTask(task_id="missing_pairs_04", train=[], test=[])
#     validator_strict_pairs = ARCTaskValidator(require_train_pairs=True, require_test_pairs=True)
#     is_valid = validator_strict_pairs.process(missing_pairs_task)
#     print(f"\nValidation for 'missing_pairs_04' (strict): {is_valid}")
#     if not is_valid: print("Errors:", "\n - ".join(validator_strict_pairs.get_errors()))

#     # Invalid task: Empty grid when not allowed
#     empty_grid_task = DummyTask(
#         task_id="empty_grid_05",
#         train=[{"input": np.array([[]]), "output": np.array([[0]])}], # Empty grid
#         test=[{"input": np.array([[1]]), "output": np.array([[0]])}]
#     )
#     validator_no_empty = ARCTaskValidator(allow_empty_grids=False)
#     is_valid = validator_no_empty.process(empty_grid_task)
#     print(f"\nValidation for 'empty_grid_05' (no empty allowed): {is_valid}")
#     if not is_valid: print("Errors:", "\n - ".join(validator_no_empty.get_errors()))

#     is_valid_empty_allowed = validator.process(empty_grid_task) # Original validator allows empty
#     print(f"Validation for 'empty_grid_05' (empty allowed): {is_valid_empty_allowed}")
#     if not is_valid_empty_allowed: print("Errors:", "\n - ".join(validator.get_errors()))
