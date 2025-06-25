from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np

from .base import BaseDataLoader, ARCTaskData # ARCTaskData is a Protocol
from arc_env.exceptions import DataLoadError # Assuming exceptions.py

# Example: A dataclass that fulfills the ARCTaskData protocol for this custom loader
from dataclasses import dataclass, field

@dataclass
class _CustomTaskData: # Implicitly fulfills ARCTaskData
    train: List[Dict[str, np.ndarray]] = field(default_factory=list)
    test: List[Dict[str, np.ndarray]] = field(default_factory=list)
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict) # Custom metadata field


class CustomDataLoader(BaseDataLoader):
    """
    Placeholder for a custom data loader.
    Users can implement this class to load ARC tasks from sources not covered
    by the standard loaders, such as:
    - A custom file format or directory structure.
    - A database.
    - A web API.
    - Procedurally generated tasks.
    """

    def __init__(self, source_identifier: Any, **kwargs: Any):
        """
        Initialize the custom data loader.

        Args:
            source_identifier: An identifier for the data source (e.g., filepath, URL, DB connection string).
            **kwargs: Additional configuration parameters for the loader.
        """
        self.source = source_identifier
        self.config_params = kwargs
        self.available_task_ids: List[str] = [] # Should be populated by initialization logic

        print(f"Initializing CustomDataLoader with source: {self.source} and params: {self.config_params}")
        # Example: Discover tasks based on source_identifier
        # self._discover_tasks()
        # For this placeholder, we'll assume no tasks are available by default.
        if not self._initialize_loader():
            print("Warning: CustomDataLoader initialization failed or found no tasks.")


    def _initialize_loader(self) -> bool:
        """
        Perform any setup required by the custom loader, like connecting to a DB,
        parsing a manifest file, or listing files in a custom directory structure.
        Should populate self.available_task_ids.
        Returns True on success, False on failure.
        """
        # --- Replace with actual custom initialization logic ---
        # Example: if source is a list of predefined tasks
        if isinstance(self.source, list) and all(isinstance(item, dict) for item in self.source):
            # Assume self.source is a list of dicts, each a task
            for i, task_dict in enumerate(self.source):
                task_id = task_dict.get("task_id", f"custom_task_{i}")
                self.available_task_ids.append(task_id)
            return True

        # Example: if source is a directory path string or Path object
        # from pathlib import Path
        # if isinstance(self.source, (str, Path)):
        #     p = Path(self.source)
        #     if p.is_dir():
        #         # self.available_task_ids = [f.stem for f in p.glob("*.mycustomformat")]
        #         pass # Implement discovery
        #     return True

        print("CustomDataLoader: _initialize_loader() needs to be implemented for the specific data source.")
        return False


    def load_task(self, task_identifier: str) -> ARCTaskData:
        """
        Loads a single ARC task specified by its identifier.

        Args:
            task_identifier: The unique identifier for the task.

        Returns:
            An object conforming to ARCTaskData protocol.

        Raises:
            DataLoadError: If the task cannot be loaded or is not found.
        """
        if task_identifier not in self.available_task_ids:
            raise DataLoadError(f"Task '{task_identifier}' not found by CustomDataLoader. Available: {self.available_task_ids}")

        # --- Replace with actual custom task loading logic ---
        # This is highly dependent on how tasks are stored and identified.
        # Example: if self.source was a list of task dicts
        if isinstance(self.source, list):
            found_task_dict = None
            for i, task_dict_item in enumerate(self.source):
                 current_id = task_dict_item.get("task_id", f"custom_task_{i}")
                 if current_id == task_identifier:
                     found_task_dict = task_dict_item
                     break

            if found_task_dict:
                # Assuming found_task_dict has 'train' and 'test' keys with grid data
                # Convert grids to numpy arrays
                try:
                    train_pairs = [
                        {"input": np.array(p["input"], dtype=np.uint8), "output": np.array(p["output"], dtype=np.uint8)}
                        for p in found_task_dict.get("train", [])
                    ]
                    test_pairs = [
                        {"input": np.array(p["input"], dtype=np.uint8), "output": np.array(p["output"], dtype=np.uint8)}
                        for p in found_task_dict.get("test", [])
                    ]
                    metadata = found_task_dict.get("metadata", {})

                    task_data = _CustomTaskData(
                        train=train_pairs,
                        test=test_pairs,
                        task_id=task_identifier,
                        metadata=metadata
                    )
                    if not self.validate_task_data(task_data, task_id_for_error=task_identifier):
                        raise DataLoadError(f"Validation failed for custom task '{task_identifier}'.")
                    return task_data # type: ignore # _CustomTaskData conforms to ARCTaskData
                except Exception as e:
                    raise DataLoadError(f"Error processing data for custom task '{task_identifier}': {e}")
            else: # Should be caught by initial check, but as a safeguard
                 raise DataLoadError(f"Task '{task_identifier}' logic error: not found in source list during load.")


        raise DataLoadError(f"CustomDataLoader: load_task() for '{task_identifier}' needs to be implemented.")
        # Example structure of what needs to be returned:
        # return _CustomTaskData(train=[...], test=[...], task_id=task_identifier, metadata={...})


    def list_available_tasks(self) -> List[str]:
        """
        Lists all task identifiers that can be loaded by this custom loader.
        This list should be populated during initialization.
        """
        return self.available_task_ids

# Example Usage:
# if __name__ == "__main__":
#     # 1. Define some dummy custom task data (as if loaded from a custom source)
#     dummy_task_list_source = [
#         {
#             "task_id": "custom_abc",
#             "train": [{"input": [[1,0],[0,1]], "output": [[0,1],[1,0]]}],
#             "test": [{"input": [[1,1],[1,1]], "output": [[0,0],[0,0]]}],
#             "metadata": {"difficulty": "easy", "source_type": "predefined_list"}
#         },
#         {
#             # No task_id, will be auto-generated
#             "train": [{"input": [[2]], "output": [[3]]}],
#             "test": [{"input": [[4]], "output": [[5]]}],
#             "metadata": {"description": "A very simple task"}
#         }
#     ]

#     try:
#         print("--- Testing CustomDataLoader ---")
#         # Initialize loader with the list of tasks as its source
#         custom_loader = CustomDataLoader(source_identifier=dummy_task_list_source)

#         available_custom_tasks = custom_loader.list_available_tasks()
#         print(f"Available custom tasks: {available_custom_tasks}")

#         if "custom_abc" in available_custom_tasks:
#             task_abc_data = custom_loader.load_task("custom_abc")
#             print("\nLoaded 'custom_abc' data:")
#             print(f"  Task ID: {task_abc_data.task_id}") # type: ignore
#             print(f"  Train input 0: \n{task_abc_data.train[0]['input']}")
#             print(f"  Test output 0: \n{task_abc_data.test[0]['output']}")
#             if hasattr(task_abc_data, 'metadata'):
#                 print(f"  Metadata: {task_abc_data.metadata}") # type: ignore

#         # Load the auto-named task (assuming it's "custom_task_1" if "custom_abc" was "custom_task_0" effectively)
#         # The auto-naming depends on the order in the list.
#         auto_named_task_id = f"custom_task_{len(dummy_task_list_source) -1}" # if custom_abc was not "custom_task_0"
#         # Let's find it more robustly
#         expected_auto_id = next((tid for tid in available_custom_tasks if tid.startswith("custom_task_")), None)

#         if expected_auto_id and expected_auto_id in available_custom_tasks:
#             task_auto_data = custom_loader.load_task(expected_auto_id)
#             print(f"\nLoaded '{expected_auto_id}' data:")
#             print(f"  Task ID: {task_auto_data.task_id}") # type: ignore
#             if hasattr(task_auto_data, 'metadata'):
#                 print(f"  Metadata: {task_auto_data.metadata}") # type: ignore
#         else:
#             print(f"\nCould not find auto-named task (expected around '{expected_auto_id}').")


#     except DataLoadError as e:
#         print(f"DataLoadError with CustomDataLoader: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

#     # Example of initializing with a non-existent path (if path loading was implemented)
#     # try:
#     #     print("\n--- Testing CustomDataLoader with invalid source ---")
#     #     invalid_loader = CustomDataLoader(source_identifier="/path/to/non_existent_custom_data_dir")
#     #     # This should print a warning or fail initialization based on _initialize_loader logic
#     # except DataLoadError as e:
#     #     print(f"Error as expected: {e}")
