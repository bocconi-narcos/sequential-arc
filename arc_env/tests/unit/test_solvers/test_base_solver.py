import pytest
from typing import Any, Dict, Optional

from arc_env.solvers.base.base_solver import BaseSolver
from arc_env.config.solver import SolverConfig # For type hints

# --- Dummy Solver Implementation for Testing BaseSolver ---
class DummyConcreteSolver(BaseSolver):
    """A concrete solver implementation for testing BaseSolver functionality."""
    def __init__(self, solver_config: Optional[SolverConfig] = None, **kwargs):
        # Convert SolverConfig object to dict for BaseSolver if that's what it expects
        # BaseSolver's __init__ takes Optional[Dict[str, Any]].
        config_dict = solver_config.to_dict() if solver_config and hasattr(solver_config, 'to_dict') else solver_config
        super().__init__(solver_config=config_dict) # type: ignore
        self.custom_param = kwargs.get("custom_param", "default_value")
        self.predict_called_count = 0
        self.reset_task_called_count = 0
        self.reset_episode_called_count = 0

    def _validate_config(self) -> None:
        super()._validate_config()
        # Example: check for a specific key if BaseSolver stored config as dict
        if isinstance(self.config, dict) and "solver_type" in self.config:
            if self.config["solver_type"] == "invalid_dummy_type":
                raise ValueError("Invalid solver_type for DummyConcreteSolver in config.")

    def predict_action(self, observation: Any, env_info: Optional[Dict[str, Any]] = None) -> Any:
        self.predict_called_count += 1
        # Dummy action, e.g., based on a config param or simple logic
        action_val = 0
        if isinstance(self.config, dict) and "default_action" in self.config:
            action_val = self.config["default_action"]
        return action_val # Return a dummy action

    def reset_for_new_task(self) -> None:
        super().reset_for_new_task()
        self.reset_task_called_count += 1
        # Reset any task-specific state here
        self.predict_called_count = 0 # Example: reset per-task call count

    def episode_reset(self) -> None:
        super().episode_reset()
        self.reset_episode_called_count += 1
        # Reset episode-specific state, could be different from task reset
        # For this dummy, let's say it doesn't reset predict_called_count.

# --- Tests for BaseSolver ---

def test_base_solver_init_no_config():
    solver = DummyConcreteSolver() # No config provided
    assert solver.config == {} # BaseSolver initializes self.config to {} if None
    assert solver.custom_param == "default_value" # Kwarg handled by DummyConcreteSolver

def test_base_solver_init_with_dict_config():
    config_dict = {"solver_id": "dummy_01", "hyperparameters": {"depth": 3}}
    solver = DummyConcreteSolver(solver_config=config_dict) # type: ignore # Passing dict
    assert solver.config == config_dict
    assert solver.config.get("solver_id") == "dummy_01"

def test_base_solver_init_with_solver_config_object():
    cfg_obj = SolverConfig(solver_id="obj_dummy_02", hyperparameters={"rate": 0.1})
    # DummyConcreteSolver converts cfg_obj to dict before passing to BaseSolver's __init__
    solver = DummyConcreteSolver(solver_config=cfg_obj)
    assert isinstance(solver.config, dict) # BaseSolver stores it as dict
    assert solver.config.get("solver_id") == "obj_dummy_02"
    assert solver.config.get("hyperparameters", {}).get("rate") == 0.1


def test_base_solver_validate_config_called_on_init():
    # _validate_config in DummyConcreteSolver raises ValueError if config has specific invalid content.
    invalid_config_dict = {"solver_type": "invalid_dummy_type"}
    with pytest.raises(ValueError, match="Invalid solver_type"):
        DummyConcreteSolver(solver_config=invalid_config_dict) # type: ignore

    valid_config_dict = {"solver_type": "valid_dummy_type"}
    try:
        DummyConcreteSolver(solver_config=valid_config_dict) # type: ignore
    except ValueError:
        pytest.fail("_validate_config raised ValueError unexpectedly for valid config.")


def test_set_current_task_context(dummy_solver_config_dict): # Using a fixture for config
    solver = DummyConcreteSolver(solver_config=dummy_solver_config_dict)

    assert solver.current_task_id is None
    assert solver.current_task_info is None
    assert solver.reset_task_called_count == 0 # Initialized to 0

    task_id1 = "task_A"
    task_info1 = {"num_train": 3, "desc": "First task"}
    solver.set_current_task_context(task_id1, task_info1)

    assert solver.current_task_id == task_id1
    assert solver.current_task_info == task_info1
    assert solver.reset_task_called_count == 1 # reset_for_new_task called

    # Call again with same task_id - should NOT call reset_for_new_task again
    solver.set_current_task_context(task_id1, {"num_train": 3, "desc": "First task (updated info)"})
    assert solver.current_task_id == task_id1 # Still same task
    assert solver.current_task_info["desc"] == "First task (updated info)" # Info updated
    assert solver.reset_task_called_count == 1 # Not incremented

    # Call with a new task_id
    task_id2 = "task_B"
    task_info2 = {"num_train": 5, "desc": "Second task"}
    solver.set_current_task_context(task_id2, task_info2)
    assert solver.current_task_id == task_id2
    assert solver.current_task_info == task_info2
    assert solver.reset_task_called_count == 2 # Incremented for new task


def test_predict_action_call(dummy_solver_config_dict):
    solver = DummyConcreteSolver(solver_config=dummy_solver_config_dict)
    assert solver.predict_called_count == 0

    # Dummy observation and info
    obs = {"grid": [[0]]}
    info = {"step": 1}

    action = solver.predict_action(obs, info)
    assert solver.predict_called_count == 1
    assert action == 0 # Default dummy action

    # Test if config affects dummy action
    config_with_action = {"default_action": 5}
    solver_custom_action = DummyConcreteSolver(solver_config=config_with_action) # type: ignore
    action2 = solver_custom_action.predict_action(obs, info)
    assert action2 == 5


def test_episode_reset_call(dummy_solver_config_dict):
    solver = DummyConcreteSolver(solver_config=dummy_solver_config_dict)
    assert solver.reset_episode_called_count == 0
    solver.episode_reset()
    assert solver.reset_episode_called_count == 1
    solver.episode_reset()
    assert solver.reset_episode_called_count == 2


def test_reset_for_new_task_resets_predict_count(dummy_solver_config_dict):
    # This tests behavior specific to DummyConcreteSolver's implementation of reset_for_new_task
    solver = DummyConcreteSolver(solver_config=dummy_solver_config_dict)
    solver.predict_action({}, {}) # predict_called_count becomes 1
    assert solver.predict_called_count == 1

    solver.set_current_task_context("new_task_for_reset_test", {}) # Calls reset_for_new_task
    assert solver.predict_called_count == 0 # DummyConcreteSolver resets this count


# --- Fixture for dummy_solver_config_dict if not in conftest.py ---
@pytest.fixture
def dummy_solver_config_dict() -> Dict[str, Any]:
    """Provides a basic solver config dictionary for tests."""
    return {
        "solver_id": "test_solver_fixture",
        "solver_type": "dummy_test_type",
        "hyperparameters": {"param1": 100}
    }

# Add tests for optional BaseSolver methods if they were implemented (train, save, load)
# e.g., test that they raise NotImplementedError by default.
# def test_base_solver_optional_methods_raise_not_implemented(dummy_solver_config_dict):
#     solver = DummyConcreteSolver(solver_config=dummy_solver_config_dict) # Dummy doesn't implement them
#     with pytest.raises(NotImplementedError):
#         solver.train(None) # type: ignore
#     with pytest.raises(NotImplementedError):
#         solver.save("path")
#     with pytest.raises(NotImplementedError):
#         solver.load("path")
