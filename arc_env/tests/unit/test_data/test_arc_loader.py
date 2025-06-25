import pytest
import numpy as np
import json
from pathlib import Path

from arc_env.data.loaders.arc_loader import ARCFileLoader, ARCCombinedFileLoader
from arc_env.data.loaders.base import ARCTaskData # Protocol
from arc_env.exceptions import DataLoadError

# Uses fixture `dummy_arc_tasks_dir` from conftest.py which creates:
# - test_task_001.json
# - test_task_002.json
# - invalid_task_format.json (missing 'test' key)
# - task_with_empty_grids.json

# --- Tests for ARCFileLoader (loads from directory of JSONs) ---

def test_arc_file_loader_init_success(dummy_arc_tasks_dir: Path):
    loader = ARCFileLoader(data_directory=dummy_arc_tasks_dir)
    assert loader.data_directory == dummy_arc_tasks_dir
    assert len(loader._task_files) >= 2 # At least the two valid tasks
    assert "test_task_001" in loader._task_files
    assert "test_task_002" in loader._task_files

def test_arc_file_loader_init_dir_not_found():
    with pytest.raises(DataLoadError, match="not found"):
        ARCFileLoader(data_directory="path_that_does_not_exist_123xyz")

def test_arc_file_loader_init_path_is_file(tmp_path: Path):
    file_path = tmp_path / "not_a_dir.json"
    file_path.touch()
    with pytest.raises(DataLoadError, match="not a directory"):
        ARCFileLoader(data_directory=file_path)

def test_arc_file_loader_list_available_tasks(dummy_arc_file_loader: ARCFileLoader):
    tasks = dummy_arc_file_loader.list_available_tasks()
    assert isinstance(tasks, list)
    # Expected files from conftest: test_task_001, test_task_002, invalid_task_format, task_with_empty_grids
    # Order should be sorted.
    expected_task_ids = sorted(["test_task_001", "test_task_002", "invalid_task_format", "task_with_empty_grids"])
    assert tasks == expected_task_ids

def test_arc_file_loader_load_task_success(dummy_arc_file_loader: ARCFileLoader):
    task_id = "test_task_001"
    task_data = dummy_arc_file_loader.load_task(task_id)

    assert task_data is not None
    assert task_data.task_id == task_id # type: ignore # _InternalARCTask has task_id
    assert len(task_data.train) == 1 # type: ignore
    assert len(task_data.test) == 1 # type: ignore

    # Check grid types and content for the first pair (based on conftest dummy data)
    # train[0]: {"input": [[1]], "output": [[0]]}
    # test[0]: {"input": [[1,2],[3,4]], "output": [[0,0],[0,0]]}
    assert isinstance(task_data.train[0]["input"], np.ndarray) # type: ignore
    assert np.array_equal(task_data.train[0]["input"], np.array([[1]], dtype=np.uint8)) # type: ignore
    assert isinstance(task_data.test[0]["output"], np.ndarray) # type: ignore
    assert np.array_equal(task_data.test[0]["output"], np.array([[0,0],[0,0]], dtype=np.uint8)) # type: ignore

def test_arc_file_loader_load_task_with_empty_grids(dummy_arc_file_loader: ARCFileLoader):
    task_id = "task_with_empty_grids" # From conftest
    task_data = dummy_arc_file_loader.load_task(task_id)
    assert task_data is not None
    # train[0]: {"input": [[]], "output": [[0]]} -> input becomes np.array([]).shape=(0,) or similar
    # test[0]: {"input": [[1]], "output": [[]]}

    # The validation in BaseDataLoader checks ndim==2. Empty list-of-list becomes 1D or 0-size 2D.
    # e.g. np.array([[]]) -> shape (1,0) if list of one empty list
    #      np.array([])   -> shape (0,)   if empty list
    # The loader's validate_task_data might need to be more lenient for empty grids,
    # or the dummy data needs to represent empty 2D grids correctly like [[],[]] for (2,0)
    # or ensure np.array([[]], dtype=np.uint8) if that's how it's stored from JSON.
    # current dummy data: "input": [[]] -> json.load gives [[]]
    # np.array([[]]) -> shape (1,0). This is 2D.
    # np.array([]) -> shape (0,). This is 1D.
    # The loader's validate_task_data should pass if ndim=2, even if one dim is 0.
    # Let's check shapes:
    assert task_data.train[0]["input"].shape == (1,0) # from json [[]]
    assert task_data.train[0]["output"].shape == (1,1) # from json [[0]]

    assert task_data.test[0]["input"].shape == (1,1) # from json [[1]]
    assert task_data.test[0]["output"].shape == (1,0) # from json [[]]

def test_arc_file_loader_load_task_not_found(dummy_arc_file_loader: ARCFileLoader):
    with pytest.raises(DataLoadError, match="not found in directory"):
        dummy_arc_file_loader.load_task("task_that_does_not_exist")

def test_arc_file_loader_load_invalid_json_format(dummy_arc_tasks_dir: Path):
    # Create a file with invalid JSON content
    invalid_json_file = dummy_arc_tasks_dir / "broken_json.json"
    with open(invalid_json_file, "w") as f:
        f.write("{'train': [[{'input': [[1]]}") # Malformed JSON

    loader = ARCFileLoader(data_directory=dummy_arc_tasks_dir) # Re-init to discover new file
    with pytest.raises(DataLoadError, match="Failed to decode JSON"):
        loader.load_task("broken_json")

def test_arc_file_loader_load_invalid_task_structure(dummy_arc_file_loader: ARCFileLoader):
    # "invalid_task_format.json" from conftest is missing the "test" key.
    # The loader's validate_task_data should catch this.
    with pytest.raises(DataLoadError, match="Validation failed for task 'invalid_task_format'"):
        dummy_arc_file_loader.load_task("invalid_task_format")


# --- Tests for ARCCombinedFileLoader (loads from master challenges.json / solutions.json) ---
# These tests need dummy challenges.json and solutions.json files.

@pytest.fixture
def dummy_combined_json_files(tmp_path: Path) -> Tuple[Path, Path]:
    challenges_content = {
        "combo_task_01": {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}] # Test output comes from solutions file
        },
        "combo_task_02": { # Task with no solution in solutions file
            "train": [{"input": [[5]], "output": [[6]]}],
            "test": [{"input": [[7]]}]
        }
    }
    solutions_content = {
        "combo_task_01": [[[4]]] # List of outputs for test cases. Here, one output for one test case.
        # combo_task_02 is missing from solutions
    }

    challenges_f = tmp_path / "challenges.json"
    solutions_f = tmp_path / "solutions.json"

    with open(challenges_f, "w") as f: json.dump(challenges_content, f)
    with open(solutions_f, "w") as f: json.dump(solutions_content, f)

    return challenges_f, solutions_f

def test_arc_combined_loader_init_success(dummy_combined_json_files: Tuple[Path, Path]):
    challenges_f, solutions_f = dummy_combined_json_files
    loader = ARCCombinedFileLoader(challenges_path=challenges_f, solutions_path=solutions_f)
    assert loader.challenges_path == challenges_f
    assert loader.solutions_path == solutions_f
    assert "combo_task_01" in loader._raw_tasks
    assert "combo_task_02" in loader._raw_tasks

def test_arc_combined_loader_list_tasks(dummy_combined_json_files: Tuple[Path, Path]):
    challenges_f, solutions_f = dummy_combined_json_files
    loader = ARCCombinedFileLoader(challenges_path=challenges_f, solutions_path=solutions_f)
    tasks = loader.list_available_tasks()
    assert sorted(tasks) == ["combo_task_01", "combo_task_02"]

def test_arc_combined_loader_load_task_with_solution(dummy_combined_json_files: Tuple[Path, Path]):
    challenges_f, solutions_f = dummy_combined_json_files
    loader = ARCCombinedFileLoader(challenges_path=challenges_f, solutions_path=solutions_f)

    task_data = loader.load_task("combo_task_01")
    assert task_data.task_id == "combo_task_01" # type: ignore
    assert len(task_data.train) == 1 # type: ignore
    assert len(task_data.test) == 1 # type: ignore
    assert np.array_equal(task_data.test[0]["input"], np.array([[3]], dtype=np.uint8)) # type: ignore
    assert np.array_equal(task_data.test[0]["output"], np.array([[4]], dtype=np.uint8)) # type: ignore Merged from solutions

def test_arc_combined_loader_load_task_missing_solution(dummy_combined_json_files: Tuple[Path, Path]):
    challenges_f, solutions_f = dummy_combined_json_files
    loader = ARCCombinedFileLoader(challenges_path=challenges_f, solutions_path=solutions_f)

    task_data = loader.load_task("combo_task_02") # This task has no entry in solutions_content
    assert task_data.task_id == "combo_task_02" # type: ignore
    assert len(task_data.test) == 1 # type: ignore
    # Test output should be a default empty grid if no solution was found
    # The loader's _load_and_prepare_data creates `np.array([[]], dtype=np.uint8)`
    expected_empty_output = np.array([[]], dtype=np.uint8)
    # np.array_equal does not work well for comparing empty arrays of different shapes like (0,) vs (1,0).
    # Let's check shape and size.
    assert task_data.test[0]["output"].shape == (1,0) or task_data.test[0]["output"].size == 0 # type: ignore

def test_arc_combined_loader_challenges_file_not_found(tmp_path: Path):
    solutions_f = tmp_path / "s.json"
    solutions_f.touch()
    with pytest.raises(DataLoadError, match="Challenges file not found"):
        ARCCombinedFileLoader("non_existent_challenges.json", solutions_f)

def test_arc_combined_loader_solutions_file_optional(tmp_path: Path):
    # Test if it handles missing solutions file gracefully (prints warning, proceeds)
    challenges_content = {"task_s_only": {"train": [], "test": [{"input": [[1]]}]}}
    challenges_f = tmp_path / "c.json"
    with open(challenges_f, "w") as f: json.dump(challenges_content, f)

    # Solutions file does not exist
    loader = ARCCombinedFileLoader(challenges_path=challenges_f, solutions_path="non_existent_solutions.json")
    assert "task_s_only" in loader.list_available_tasks()
    task_data = loader.load_task("task_s_only")
    assert task_data.test[0]["output"].size == 0 # type: ignore # Default empty output


# Add more tests:
# - Corrupted content in challenges.json or solutions.json for ARCCombinedFileLoader.
# - Edge cases for grid structures (e.g. non-rectangular lists of lists if not caught by JSON).
# - Behavior of BaseDataLoader.validate_task_data with various inputs.
# - get_all_tasks method.
