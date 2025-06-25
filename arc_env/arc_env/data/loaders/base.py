from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Protocol, runtime_checkable # Tuple not directly used in signatures
from pathlib import Path
import numpy as np

# Define a type for a single ARC task/challenge.
# This usually includes training pairs and test pairs.
# Each pair has an 'input' and 'output' grid.
class ARCTaskData(Protocol):
    train: List[Dict[str, np.ndarray]] # List of {"input": grid, "output": grid}
    test: List[Dict[str, np.ndarray]]  # List of {"input": grid, "output": grid}
    task_id: Optional[str] # Optional identifier for the task

@runtime_checkable
class BaseDataLoader(Protocol): # Using Protocol for flexibility, can also be ABC
    """
    Abstract base class (or Protocol) for data loaders.

    Data loaders are responsible for fetching ARC task data from various sources
    (e.g., local files, databases, web URLs).
    """

    @abstractmethod
    def load_task(self, task_identifier: Any) -> ARCTaskData:
        """
        Loads a single ARC task specified by an identifier.
        The nature of `task_identifier` (e.g., string ID, filepath, index)
        depends on the concrete loader implementation.

        Args:
            task_identifier: Identifier for the task to load.

        Returns:
            An ARCTaskData object containing the task's training and test pairs.

        Raises:
            DataLoadError: If the task cannot be loaded (e.g., not found, corrupted).
            NotImplementedError: If the method is not implemented by a subclass.
        """
        ...

    @abstractmethod
    def list_available_tasks(self) -> List[Any]:
        """
        Lists all available task identifiers that can be loaded by this loader.

        Returns:
            A list of task identifiers.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        ...

    # Optional methods that might be useful:
    # def get_all_tasks(self) -> Dict[Any, ARCTaskData]:
    #     """Loads all available tasks into a dictionary."""
    #     all_tasks = {}
    #     for task_id in self.list_available_tasks():
    #         try:
    #             all_tasks[task_id] = self.load_task(task_id)
    #         except Exception as e: # Catch DataLoadError specifically if defined
    #             print(f"Warning: Failed to load task {task_id} while getting all tasks: {e}")
    #     return all_tasks

    # def validate_task_data(self, task_data: ARCTaskData) -> bool:
    #     """
    #     Performs basic validation on the structure of loaded task data.
    #     Checks for presence of 'train'/'test' keys, and that grids are numpy arrays.
    #     """
    #     if not hasattr(task_data, 'train') or not hasattr(task_data, 'test'):
    #         return False
    #     if not isinstance(task_data.train, list) or not isinstance(task_data.test, list):
    #         return False

    #     for pair_list_name, pair_list in [("train", task_data.train), ("test", task_data.test)]:
    #         for i, pair in enumerate(pair_list):
    #             if not isinstance(pair, dict): return False
    #             if "input" not in pair or "output" not in pair: return False
    #             if not isinstance(pair["input"], np.ndarray) or not isinstance(pair["output"], np.ndarray):
    #                 return False
    #             if pair["input"].ndim != 2 or pair["output"].ndim != 2:
    #                 # ARC grids are 2D, could add color range checks too
    #                 print(f"Warning: Grid in {pair_list_name} pair {i} is not 2D.")
    #                 return False
    #     return True


# Example of a concrete ARCTaskData implementation (though Protocol is often enough)
# from dataclasses import dataclass, field

# @dataclass
# class ConcreteARCTaskData:
#     train: List[Dict[str, np.ndarray]] = field(default_factory=list)
#     test: List[Dict[str, np.ndarray]] = field(default_factory=list)
#     task_id: Optional[str] = None

#     def __post_init__(self):
#         # Could add validation here using a method like the commented out one above
#         pass

# This BaseDataLoader uses Protocol for more flexibility, allowing various classes
# to act as data loaders without explicit inheritance, as long as they provide
# the required methods. If strict inheritance is preferred, change to `class BaseDataLoader(ABC):`
# and ensure subclasses inherit from it.
# For now, the skeleton provided `ARCDataLoader(BaseDataLoader)` implies ABC style was intended.
# Let's switch to ABC for consistency with the provided ARCDataLoader skeleton.

del BaseDataLoader # Remove the Protocol definition

class BaseDataLoaderABC(ABC): # Renamed to avoid conflict if user runs this file directly with old name
    """
    Abstract base class for data loaders.

    Data loaders are responsible for fetching ARC task data from various sources
    (e.g., local files, databases, web URLs).
    """

    @abstractmethod
    def load_task(self, task_identifier: Any) -> ARCTaskData: # ARCTaskData is still a Protocol
        """
        Loads a single ARC task specified by an identifier.
        The nature of `task_identifier` (e.g., string ID, filepath, index)
        depends on the concrete loader implementation.

        Args:
            task_identifier: Identifier for the task to load.

        Returns:
            An ARCTaskData object containing the task's training and test pairs.

        Raises:
            DataLoadError: If the task cannot be loaded (e.g., not found, corrupted).
        """
        pass

    @abstractmethod
    def list_available_tasks(self) -> List[Any]:
        """
        Lists all available task identifiers that can be loaded by this loader.

        Returns:
            A list of task identifiers.
        """
        pass

    def get_all_tasks(self) -> Dict[Any, ARCTaskData]:
        """Loads all available tasks into a dictionary. Can be overridden for efficiency."""
        all_tasks = {}
        for task_id in self.list_available_tasks():
            try:
                all_tasks[task_id] = self.load_task(task_id)
            except Exception as e: # Catch DataLoadError specifically if defined by exceptions.py
                from arc_env.exceptions import DataLoadError # Local import to avoid circularity at module load
                if isinstance(e, DataLoadError):
                    print(f"Warning: Failed to load task {task_id} (DataLoadError): {e}")
                else:
                    print(f"Warning: Failed to load task {task_id} (Unknown Error): {e}")
        return all_tasks

    def validate_task_data(self, task_data: ARCTaskData, task_id_for_error: Any = "Unknown") -> bool:
        """
        Performs basic validation on the structure of loaded task data.
        Checks for presence of 'train'/'test' keys, and that grids are numpy arrays.
        """
        if not hasattr(task_data, 'train') or not isinstance(task_data.train, list): # type: ignore
            print(f"Validation Error (Task {task_id_for_error}): 'train' field missing or not a list.")
            return False
        if not hasattr(task_data, 'test') or not isinstance(task_data.test, list): # type: ignore
            print(f"Validation Error (Task {task_id_for_error}): 'test' field missing or not a list.")
            return False

        for pair_list_name, pair_list in [("train", task_data.train), ("test", task_data.test)]: # type: ignore
            if not pair_list and pair_list_name == "test": # Test list must not be empty
                print(f"Validation Error (Task {task_id_for_error}): 'test' pair list is empty.")
                return False

            for i, pair in enumerate(pair_list):
                if not isinstance(pair, dict):
                    print(f"Validation Error (Task {task_id_for_error}): {pair_list_name} pair {i} is not a dict.")
                    return False
                if "input" not in pair or "output" not in pair:
                    print(f"Validation Error (Task {task_id_for_error}): {pair_list_name} pair {i} missing 'input' or 'output'.")
                    return False

                for grid_key in ["input", "output"]:
                    grid = pair[grid_key]
                    if not isinstance(grid, np.ndarray):
                        print(f"Validation Error (Task {task_id_for_error}): {pair_list_name} pair {i} '{grid_key}' is not a numpy array.")
                        return False
                    if grid.ndim != 2:
                        print(f"Validation Error (Task {task_id_for_error}): {pair_list_name} pair {i} '{grid_key}' is not 2D (shape: {grid.shape}).")
                        return False
                    if grid.size == 0: # Empty grid (e.g. shape (0,5) or (5,0) or (0,0))
                         # ARC allows empty grids e.g. task 00576224 test output
                         # print(f"Warning (Task {task_id_for_error}): {pair_list_name} pair {i} '{grid_key}' has zero size (shape: {grid.shape}).")
                         pass # Allow empty grids for now.
                    # Could add color range checks, e.g., np.all((grid >= 0) & (grid <= 9))
        return True

# Make BaseDataLoader an alias to the ABC version for consistency with skeleton
BaseDataLoader = BaseDataLoaderABC
