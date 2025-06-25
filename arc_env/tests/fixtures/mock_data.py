import numpy as np
from typing import List, Dict, Any, Optional

from arc_env.data.loaders.base import ARCTaskData # Protocol
from arc_env.config.environment import EnvironmentConfig
from arc_env.config.action_space import ActionSpaceConfig
from arc_env.config.solver import SolverConfig

# This file provides mock data structures and objects that can be used as fixtures in tests.
# This is different from sample_grids.py, which focuses on np.ndarray grids.
# Here, we can define mock ARCTaskData objects, mock configs, etc.

# --- Mock ARCTaskData ---
class MockARCTask(ARCTaskData): # type: ignore # Protocol can be "implemented" by concrete class
    """A concrete class implementing the ARCTaskData protocol for mocking."""
    def __init__(
        self,
        task_id: str,
        train_pairs: List[Dict[str, np.ndarray]],
        test_pairs: List[Dict[str, np.ndarray]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.task_id = task_id
        self.train = train_pairs
        self.test = test_pairs
        self.metadata = metadata if metadata is not None else {} # Custom field for mocks

# Example mock task data instances
MOCK_TASK_01_DATA = MockARCTask(
    task_id="mock_task_01",
    train=[
        {"input": np.array([[1,0],[0,1]]), "output": np.array([[0,1],[1,0]])}
    ],
    test=[
        {"input": np.array([[2,2],[2,2]]), "output": np.array([[3,3],[3,3]])}
    ],
    metadata={"description": "A simple mock task for testing."}
)

MOCK_TASK_02_DATA_NO_TRAIN = MockARCTask(
    task_id="mock_task_02_no_train",
    train=[], # No training pairs
    test=[
        {"input": np.array([[5]]), "output": np.array([[6]])}
    ],
    metadata={"source": "generated_mock"}
)

MOCK_TASK_03_DATA_MULTIPLE_TEST = MockARCTask(
    task_id="mock_task_03_multi_test",
    train=[
        {"input": np.array([[1,1],[1,1]]), "output": np.array([[0,0],[0,0]])}
    ],
    test=[
        {"input": np.array([[1,0]]), "output": np.array([[0,1]])},
        {"input": np.array([[0,1]]), "output": np.array([[1,0]])}
    ]
)

ALL_MOCK_TASKS: Dict[str, MockARCTask] = {
    "mock_task_01": MOCK_TASK_01_DATA,
    "mock_task_02_no_train": MOCK_TASK_02_DATA_NO_TRAIN,
    "mock_task_03_multi_test": MOCK_TASK_03_DATA_MULTIPLE_TEST,
}

def get_mock_task_data(task_id: str) -> Optional[MockARCTask]:
    """Returns a copy of a mock task data object."""
    import copy # For deep copy if tasks might be modified by tests
    task = ALL_MOCK_TASKS.get(task_id)
    return copy.deepcopy(task) if task else None


# --- Mock Configurations ---
MOCK_ENV_CONFIG_SMALL = EnvironmentConfig(
    canvas_size=8,
    max_steps=15,
    step_penalty=-0.05,
    completion_bonus=15.0
)

MOCK_ACTION_CONFIG_MINIMAL_JOINT = ActionSpaceConfig(
    mode="joint",
    preset="minimal", # Assumes "minimal" preset is known/loadable by registry in tests
    available_presets=["default", "minimal", "custom_test_preset"]
)

MOCK_SOLVER_CONFIG_HEURISTIC = SolverConfig( # Using base SolverConfig
    solver_type="mock_heuristic_type",
    solver_id="mock_h_solver_001",
    hyperparameters={"depth_limit": 5, "strategy": "bfs"}
)

# --- Mock Objects (e.g., mock data loader) ---
class MockDataLoader: # Does not need to inherit BaseDataLoader if only used for mocking specific tests
    def __init__(self, tasks_dict: Optional[Dict[str, ARCTaskData]] = None):
        self.tasks = tasks_dict if tasks_dict is not None else {}

    def load_task(self, task_identifier: Any) -> ARCTaskData:
        if task_identifier in self.tasks:
            return self.tasks[task_identifier]
        from arc_env.exceptions import DataLoadError # Local import
        raise DataLoadError(f"MockDataLoader: Task '{task_identifier}' not found.")

    def list_available_tasks(self) -> List[Any]:
        return list(self.tasks.keys())

    # Add validate_task_data if needed for full BaseDataLoader compatibility in tests
    def validate_task_data(self, task_data: ARCTaskData, task_id_for_error: Any = "Unknown") -> bool:
        # Basic pass-through validation for mock
        return True if task_data.train is not None and task_data.test is not None else False # type: ignore

    def get_all_tasks(self) -> Dict[Any, ARCTaskData]:
        return self.tasks.copy()


def get_mock_data_loader_with_defaults() -> MockDataLoader:
    """Returns a MockDataLoader populated with the default mock tasks."""
    return MockDataLoader(tasks_dict=ALL_MOCK_TASKS) # type: ignore


if __name__ == '__main__':
    print("Mock Data Fixtures:")

    task1 = get_mock_task_data("mock_task_01")
    if task1:
        print(f"\nMock Task ID: {task1.task_id}")
        print(f"  Description: {task1.metadata.get('description')}") # type: ignore
        print(f"  Train pairs: {len(task1.train)}")
        print(f"  Test pairs: {len(task1.test)}")
        print(f"  First train input:\n{task1.train[0]['input']}")

    print(f"\nMock Env Config (Small): canvas_size={MOCK_ENV_CONFIG_SMALL.canvas_size}")
    print(f"Mock Action Config (Minimal Joint): mode='{MOCK_ACTION_CONFIG_MINIMAL_JOINT.mode}', preset='{MOCK_ACTION_CONFIG_MINIMAL_JOINT.preset}'")
    print(f"Mock Solver Config (Heuristic): type='{MOCK_SOLVER_CONFIG_HEURISTIC.solver_type}', params={MOCK_SOLVER_CONFIG_HEURISTIC.hyperparameters}")

    mock_loader = get_mock_data_loader_with_defaults()
    print(f"\nMock Data Loader available tasks: {mock_loader.list_available_tasks()}")
    loaded_task_via_mock = mock_loader.load_task("mock_task_02_no_train")
    print(f"Loaded via MockLoader: {loaded_task_via_mock.task_id}, Train pairs: {len(loaded_task_via_mock.train)}") # type: ignore
