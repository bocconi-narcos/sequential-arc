from __future__ import annotations # For forward references like Path

from typing import Dict, Any, List, Tuple, cast, Union, Optional # Added Union, Optional
from pathlib import Path
import json
import numpy as np

from .base import BaseDataLoader, ARCTaskData # ARCTaskData is a Protocol
from arc_env.exceptions import DataLoadError # Assuming exceptions.py will be/is created

# Define a concrete dataclass that implements the ARCTaskData protocol for internal use.
from dataclasses import dataclass, field

@dataclass
class _InternalARCTask: # This class implicitly fulfills ARCTaskData protocol.
    """Concrete implementation of ARCTaskData for ARCFileLoader."""
    train: List[Dict[str, np.ndarray]] = field(default_factory=list)
    test: List[Dict[str, np.ndarray]] = field(default_factory=list)
    task_id: Optional[str] = None # Store the task ID (filename without .json)

class ARCFileLoader(BaseDataLoader):
    """
    Robust ARC dataset loader for loading tasks from individual JSON files in a directory,
    as found in the original ARC dataset structure (e.g., 'training/*.json', 'evaluation/*.json').

    This loader expects a directory path containing multiple .json files, each representing one task.
    """

    def __init__(self, data_directory: Union[str, Path]) -> None:
        """
        Args:
            data_directory: Path to the directory containing ARC task JSON files.
        """
        self.data_directory = Path(data_directory)
        self._validate_directory()
        self._task_files: Dict[str, Path] = self._discover_task_files()

    def _validate_directory(self) -> None:
        """Validate that the data directory exists and is a directory."""
        if not self.data_directory.exists():
            raise DataLoadError(f"Data directory not found: {self.data_directory}")
        if not self.data_directory.is_dir():
            raise DataLoadError(f"Data path is not a directory: {self.data_directory}")

    def _discover_task_files(self) -> Dict[str, Path]:
        """Scans the data directory for .json files and maps task IDs to file paths."""
        task_files = {}
        for filepath in self.data_directory.glob("*.json"):
            if filepath.is_file():
                task_id = filepath.stem # Filename without extension
                task_files[task_id] = filepath
        if not task_files:
            print(f"Warning: No .json task files found in directory: {self.data_directory}")
        return task_files

    def list_available_tasks(self) -> List[str]:
        """Lists task IDs (filenames without .json) available in the directory."""
        return sorted(list(self._task_files.keys()))

    def load_task(self, task_identifier: str) -> ARCTaskData:
        """
        Loads a single ARC task from its JSON file.

        Args:
            task_identifier: The task ID (filename without .json suffix).

        Returns:
            An _InternalARCTask object (which conforms to ARCTaskData protocol)
            containing the task's data.

        Raises:
            DataLoadError: If the task file is not found, is not valid JSON,
                           or does not conform to the expected ARC task structure.
        """
        if task_identifier not in self._task_files:
            raise DataLoadError(f"Task '{task_identifier}' not found in directory {self.data_directory}. "
                                f"Available tasks: {self.list_available_tasks()}")

        filepath = self._task_files[task_identifier]

        try:
            with open(filepath, 'r') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise DataLoadError(f"Failed to decode JSON for task '{task_identifier}' from {filepath}: {e}")
        except IOError as e:
            raise DataLoadError(f"Failed to read file for task '{task_identifier}' from {filepath}: {e}")

        # Convert lists of lists into numpy arrays for train and test pairs
        try:
            train_pairs = [
                {"input": np.array(pair["input"], dtype=np.uint8),
                 "output": np.array(pair["output"], dtype=np.uint8)}
                for pair in raw_data.get("train", [])
            ]
            test_pairs = [
                {"input": np.array(pair["input"], dtype=np.uint8),
                 "output": np.array(pair["output"], dtype=np.uint8)}
                for pair in raw_data.get("test", [])
            ]
        except Exception as e: # Catch errors during np.array conversion or if structure is wrong
            raise DataLoadError(f"Error converting grids to numpy arrays for task '{task_identifier}': {e}. "
                                "Ensure 'input'/'output' are valid grid structures (list of lists of int).")

        task_data_obj = _InternalARCTask(train=train_pairs, test=test_pairs, task_id=task_identifier)

        # Validate the loaded data structure (inherited from BaseDataLoader)
        if not self.validate_task_data(task_data_obj, task_id_for_error=task_identifier):
            # validate_task_data prints its own detailed errors.
            raise DataLoadError(f"Validation failed for task '{task_identifier}'. Check console for details.")

        return task_data_obj


# The skeleton provided a different ARCDataLoader that loads from two master files.
# Let's call that ARCCombinedFileLoader to differentiate.
# The current ARCFileLoader loads from a directory of individual task files.

class ARCCombinedFileLoader(BaseDataLoader):
    """
    Robust ARC dataset loader with validation, loading from combined
    challenges and solutions JSON files.
    This matches the structure of the original skeleton for ARCDataLoader.
    """

    def __init__(self, challenges_path: Union[str, Path], solutions_path: Union[str, Path]) -> None:
        self.challenges_path = Path(challenges_path)
        self.solutions_path = Path(solutions_path)
        self._validate_paths()

        self._challenges_data: Dict[str, Any] = {} # Stores raw data from challenges.json
        self._solutions_data: Dict[str, Any] = {} # Stores raw data from solutions.json (if used directly)
                                                  # Or this loader could combine them into ARCTaskData.

        self._raw_tasks: Dict[str, Dict[str, Any]] = {} # Stores task data keyed by task_id
        self._load_and_prepare_data()


    def _validate_paths(self) -> None:
        """Validate file paths exist."""
        if not self.challenges_path.exists():
            raise DataLoadError(f"Challenges file not found: {self.challenges_path}")
        if not self.solutions_path.exists():
            # Solutions might be optional or integrated differently depending on dataset format.
            # For now, let's assume it's required if provided in init.
            print(f"Warning: Solutions file '{self.solutions_path}' not found. Some operations or full task data might be unavailable.")
            # raise DataLoadError(f"Solutions file not found: {self.solutions_path}") # Original skeleton was stricter

    def _load_and_prepare_data(self) -> None:
        """Load and validate ARC data from the combined files."""
        try:
            with open(self.challenges_path, 'r') as f:
                self._challenges_data = json.load(f) # This is expected to be a dict: {"task_id": task_json_content, ...}
        except (json.JSONDecodeError, IOError) as e:
            raise DataLoadError(f"Failed to load or parse challenges data from {self.challenges_path}: {e}")

        if self.solutions_path.exists(): # Only load solutions if path is valid
            try:
                with open(self.solutions_path, 'r') as f:
                    self._solutions_data = json.load(f) # Also a dict: {"task_id": solution_json_content, ...}
            except (json.JSONDecodeError, IOError) as e:
                # Non-critical if solutions are optional, but log a warning.
                print(f"Warning: Failed to load or parse solutions data from {self.solutions_path}: {e}")
                self._solutions_data = {} # Ensure it's an empty dict if loading fails
        else:
            self._solutions_data = {} # No solutions file provided or found

        # Combine challenges and solutions data if necessary, or prepare for load_task.
        # The original ARC dataset format has train and test pairs in the same file.
        # If challenges.json contains the full task data (train+test inputs, train outputs)
        # and solutions.json contains test outputs, we need to merge them.
        #
        # Assuming challenges.json has format: { "task_id1": {"train": [...], "test": [{"input": ...}]} }
        # And solutions.json has format:     { "task_id1": [{"output": ...}, ...] } for test outputs.
        # This requires careful merging.

        # For now, let's assume self._challenges_data contains the full task structure
        # as expected by ARCTaskData, but grids are lists of lists.
        # And self._solutions_data provides the 'output' for the 'test' pairs if not in challenges.

        for task_id, task_content in self._challenges_data.items():
            if not isinstance(task_content, dict):
                print(f"Warning: Task content for '{task_id}' in {self.challenges_path} is not a dictionary. Skipping.")
                continue

            current_task_data: Dict[str, Any] = {"train": [], "test": []}

            # Process training pairs
            raw_train_pairs = task_content.get("train", [])
            if not isinstance(raw_train_pairs, list):
                print(f"Warning: 'train' data for task '{task_id}' is not a list. Skipping train pairs.")
                raw_train_pairs = []

            for i, pair_data in enumerate(raw_train_pairs):
                if not isinstance(pair_data, dict) or "input" not in pair_data or "output" not in pair_data:
                    print(f"Warning: Invalid train pair structure for task '{task_id}', pair {i}. Skipping.")
                    continue
                current_task_data["train"].append({
                    "input": np.array(pair_data["input"], dtype=np.uint8),
                    "output": np.array(pair_data["output"], dtype=np.uint8)
                })

            # Process test pairs (inputs from challenges, outputs from solutions)
            raw_test_inputs = task_content.get("test", [])
            if not isinstance(raw_test_inputs, list):
                print(f"Warning: 'test' data for task '{task_id}' is not a list. Skipping test pairs.")
                raw_test_inputs = []

            task_solutions = self._solutions_data.get(task_id, []) # List of test outputs for this task
            if not isinstance(task_solutions, list):
                 print(f"Warning: Solutions for task '{task_id}' in {self.solutions_path} is not a list. Assuming no solutions.")
                 task_solutions = []


            for i, pair_data in enumerate(raw_test_inputs):
                if not isinstance(pair_data, dict) or "input" not in pair_data:
                    print(f"Warning: Invalid test input structure for task '{task_id}', pair {i}. Skipping.")
                    continue

                test_output_grid = np.array([[]], dtype=np.uint8) # Default empty grid if no solution
                if i < len(task_solutions):
                    # Original ARC format for solutions.json might be just a list of output grids per task.
                    # Or it might be [{"solution_1": grid}, {"solution_2": grid}]
                    # Or {"task_id": [output_grid_1, output_grid_2]}
                    # The skeleton's solutions_path implies a file that might map task_id to its solutions.
                    # Let's assume task_solutions[i] is the output grid for test_input[i].
                    # This needs to match the actual format of solutions.json.
                    # If solutions_data[task_id] is a list of output grids:
                    solution_for_pair = task_solutions[i]
                    if isinstance(solution_for_pair, list): # If it's a grid
                        test_output_grid = np.array(solution_for_pair, dtype=np.uint8)
                    elif isinstance(solution_for_pair, dict) and "output" in solution_for_pair: # If it's {"output": grid}
                        test_output_grid = np.array(solution_for_pair["output"], dtype=np.uint8)
                    else:
                        print(f"Warning: Unrecognized solution format for task '{task_id}', test pair {i}. Using empty output.")
                else:
                    if self.solutions_path.exists(): # Only warn if solutions were expected
                        print(f"Warning: No solution found for task '{task_id}', test pair {i}. Using empty output.")

                current_task_data["test"].append({
                    "input": np.array(pair_data["input"], dtype=np.uint8),
                    "output": test_output_grid
                })

            self._raw_tasks[task_id] = current_task_data

        if not self._raw_tasks:
            raise DataLoadError("No tasks could be prepared from the provided challenge/solution files.")


    def list_available_tasks(self) -> List[str]:
        """Lists task IDs available from the loaded challenge data."""
        return sorted(list(self._raw_tasks.keys()))

    def load_task(self, task_identifier: str) -> ARCTaskData:
        """
        Retrieves a prepared ARC task by its ID.
        """
        if task_identifier not in self._raw_tasks:
            raise DataLoadError(f"Task '{task_identifier}' not found in loaded data. Available: {self.list_available_tasks()}")

        task_raw_data = self._raw_tasks[task_identifier]

        # Construct the ARCTaskData object (using _InternalARCTask for concrete type)
        task_data_obj = _InternalARCTask(
            train=task_raw_data["train"],
            test=task_raw_data["test"],
            task_id=task_identifier
        )

        if not self.validate_task_data(task_data_obj, task_id_for_error=task_identifier):
            raise DataLoadError(f"Validation failed for task '{task_identifier}' from combined files.")

        return task_data_obj


# Example Usage (requires dummy files challenges.json, solutions.json or a directory with task files)
# if __name__ == "__main__":
#     # Create dummy files for ARCCombinedFileLoader
#     dummy_challenges_content = {
#         "task001": {
#             "train": [{"input": [[1]], "output": [[2]]}],
#             "test": [{"input": [[3]]}]
#         },
#         "task002": {
#             "train": [{"input": [[0]], "output": [[1]]}],
#             "test": [{"input": [[8]], "input_2": [[9]]}] # Test multiple test inputs per task
#         }
#     }
#     dummy_solutions_content = { # Solutions for the *test* inputs of tasks
#         "task001": [[[4]]], # Solution for test input [[3]] is [[4]]
#         "task002": [[[7]]]  # Solution for test input [[8]] is [[7]]. Assumes solutions match order of test inputs.
#                            # If task002 had two test inputs, this list would have two solution grids.
#     }

#     challenges_file = Path("dummy_challenges.json")
#     solutions_file = Path("dummy_solutions.json")

#     with open(challenges_file, "w") as f: json.dump(dummy_challenges_content, f)
#     with open(solutions_file, "w") as f: json.dump(dummy_solutions_content, f)

#     try:
#         print("--- Testing ARCCombinedFileLoader ---")
#         combined_loader = ARCCombinedFileLoader(challenges_file, solutions_file)
#         print("Available tasks (combined):", combined_loader.list_available_tasks())
#         if "task001" in combined_loader.list_available_tasks():
#             task1_data = combined_loader.load_task("task001")
#             print("Task 001 data (combined):")
#             print("  Train input:", task1_data.train[0]["input"])
#             print("  Train output:", task1_data.train[0]["output"])
#             print("  Test input:", task1_data.test[0]["input"])
#             print("  Test output:", task1_data.test[0]["output"]) # Should be [[4]]
#         if "task002" in combined_loader.list_available_tasks():
#             task2_data = combined_loader.load_task("task002")
#             print("Task 002 Test Output:", task2_data.test[0]["output"]) # Should be [[7]]


#     except DataLoadError as e:
#         print(f"DataLoadError (CombinedLoader): {e}")
#     finally:
#         challenges_file.unlink(missing_ok=True)
#         solutions_file.unlink(missing_ok=True)

#     # Create dummy files for ARCFileLoader
#     dummy_task_dir = Path("dummy_arc_tasks")
#     dummy_task_dir.mkdir(exist_ok=True)
#     task_a_content = {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3]], "output": [[4]]}]}
#     task_b_content = {"train": [{"input": [[5]], "output": [[6]]}], "test": [{"input": [[7]], "output": [[8]]}]}
#     with open(dummy_task_dir / "task_a.json", "w") as f: json.dump(task_a_content, f)
#     with open(dummy_task_dir / "task_b.json", "w") as f: json.dump(task_b_content, f)

#     try:
#         print("\n--- Testing ARCFileLoader ---")
#         file_loader = ARCFileLoader(dummy_task_dir)
#         print("Available tasks (file loader):", file_loader.list_available_tasks())
#         if "task_a" in file_loader.list_available_tasks():
#             task_a_data = file_loader.load_task("task_a")
#             print("Task A data (file loader):")
#             print("  Train input:", task_a_data.train[0]["input"])
#             print("  Test output:", task_a_data.test[0]["output"])
#     except DataLoadError as e:
#         print(f"DataLoadError (FileLoader): {e}")
#     finally:
#         import shutil
#         shutil.rmtree(dummy_task_dir, ignore_errors=True)
