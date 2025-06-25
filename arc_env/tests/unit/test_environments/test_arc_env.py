import pytest
import numpy as np
from gymnasium.spaces import Discrete, Dict as GymDict

from arc_env.environments.arc_env import ARCEnv
from arc_env.config.environment import EnvironmentConfig
from arc_env.config.action_space import ActionSpaceConfig
# from arc_env.data.loaders.arc_loader import ARCFileLoader # Loaded via fixture
# from arc_env.dsl.core.operation_registry import OperationRegistry # Loaded via fixture
from arc_env.exceptions import TaskNotSetError

# Uses fixtures from conftest.py:
# - dummy_env_config
# - dummy_action_space_config
# - dummy_arc_file_loader
# - dummy_operation_registry
# - basic_arc_env

def test_arc_env_creation(basic_arc_env: ARCEnv):
    """Test if the ARCEnv can be created with basic configurations."""
    assert basic_arc_env is not None, "basic_arc_env fixture failed to provide an environment."
    assert isinstance(basic_arc_env, ARCEnv)
    assert basic_arc_env.current_task_id == "test_task_001", "Initial task not set as expected."

def test_arc_env_set_task(basic_arc_env: ARCEnv):
    """Test the set_task method."""
    assert basic_arc_env is not None
    new_task_id = "test_task_002"
    basic_arc_env.set_task(new_task_id)
    assert basic_arc_env.current_task_id == new_task_id
    assert basic_arc_env.current_task_data is not None
    assert len(basic_arc_env.current_task_data.train) > 0 # Assuming test_task_002 has train data

    with pytest.raises(TaskNotSetError): # Or DataLoadError depending on loader
        basic_arc_env.set_task("non_existent_task_id_xyz")

def test_arc_env_reset(basic_arc_env: ARCEnv):
    """Test the reset method."""
    assert basic_arc_env is not None
    basic_arc_env.set_task("test_task_001") # Ensure a known task is set

    obs, info = basic_arc_env.reset()

    assert isinstance(obs, dict), "Observation should be a dictionary."
    assert "task_grid" in obs
    assert "train_pairs" in obs
    assert "test_input_grid" in obs

    # Check shapes based on dummy_env_config (canvas_size=10)
    canvas_size = basic_arc_env.env_config.canvas_size
    assert obs["task_grid"].shape == (canvas_size, canvas_size)
    assert obs["test_input_grid"].shape == (canvas_size, canvas_size)

    # Check train_pairs structure (ARCObservationSpace pads to max_train_pairs)
    max_pairs = basic_arc_env.observation_space.max_train_pairs
    assert isinstance(obs["train_pairs"], tuple) # ARCObservationSpace returns tuple for Tuple space
    assert len(obs["train_pairs"]) == max_pairs
    if max_pairs > 0:
        first_pair = obs["train_pairs"][0]
        assert isinstance(first_pair, dict)
        assert "input" in first_pair and "output" in first_pair
        assert first_pair["input"].shape == (canvas_size, canvas_size)
        assert first_pair["output"].shape == (canvas_size, canvas_size)

    assert isinstance(info, dict)
    assert info.get("task_id") == "test_task_001"
    assert info.get("steps_taken_in_episode") == 0

    # Test reset with options to change task
    obs2, info2 = basic_arc_env.reset(options={"task_id": "test_task_002"})
    assert info2.get("task_id") == "test_task_002"
    # Check if current_grid reflects the new task's first test input
    # This requires knowing the content of test_task_002's first test input.
    # For dummy_arc_tasks_dir, test_task_002 test input is [[7]]
    # Expected: canvas_size=10, so [[7,0,0...],[0,0,0...],...]
    expected_grid_val_for_task002 = 7 # From conftest dummy task
    assert obs2["task_grid"][0,0] == expected_grid_val_for_task002


def test_arc_env_step(basic_arc_env: ARCEnv):
    """Test the step method with a sample action."""
    assert basic_arc_env is not None
    basic_arc_env.set_task("test_task_001")
    initial_obs, _ = basic_arc_env.reset()

    # Sample a valid action from the environment's action space
    action = basic_arc_env.action_space.sample()

    obs, reward, terminated, truncated, info = basic_arc_env.step(action)

    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert info.get("steps_taken_in_episode") == 1
    assert "action_decoded_str" in info

    # More specific checks for reward, termination based on action would require
    # knowing the effect of the sampled action and the task's solution.
    # For now, just check types and basic info.

    # Example: Test for termination if task is solved
    # This requires an action that solves test_task_001.
    # test_task_001: test input [[1,2],[3,4]], output [[0,0],[0,0]] (canvas 10x10)
    # A "fill with 0" action on "select_all" should solve it if these ops exist.

    # Try to find "select_all" and "fill_selected" (with color 0)
    # This depends on "default" preset loaded by dummy_operation_registry
    select_all_idx = -1
    fill_0_idx = -1

    if "selection" in basic_arc_env.action_space._category_keys:
        sel_cat_idx = basic_arc_env.action_space._category_keys.index("selection")
        for i, op in enumerate(basic_arc_env.action_space._op_categories["selection"]):
            if op.to_string().lower() == "selectall()": # Brittle check
                select_all_idx = i
                break

    if "color" in basic_arc_env.action_space._category_keys:
        col_cat_idx = basic_arc_env.action_space._category_keys.index("color")
        for i, op in enumerate(basic_arc_env.action_space._op_categories["color"]):
            if op.to_string().lower() == "fill(color=0)": # Brittle
                 # Check if FillOperation with color=0 is available.
                 # The default preset might have fill_selected with color=1.
                 # Let's assume we can find an action that fills with 0.
                 # This test is becoming too dependent on preset details.
                pass # Cannot reliably find fill_0_idx without knowing preset structure.


    # If we had a known solving action:
    # solving_action = ...
    # basic_arc_env.reset() # Reset to start of task
    # obs, reward, terminated, truncated, info = basic_arc_env.step(solving_action)
    # assert terminated is True
    # assert reward >= basic_arc_env.env_config.completion_bonus (approx)


def test_arc_env_render_modes(basic_arc_env: ARCEnv):
    """Test if render modes run without error."""
    assert basic_arc_env is not None
    basic_arc_env.set_task("test_task_001")
    basic_arc_env.reset()

    original_mode = basic_arc_env.render_mode

    try:
        basic_arc_env.render_mode = "ansi"
        ansi_out = basic_arc_env.render()
        assert isinstance(ansi_out, str) # Current _render_ansi_str returns str

        basic_arc_env.render_mode = "rgb_array"
        rgb_out = basic_arc_env.render()
        assert isinstance(rgb_out, np.ndarray)
        assert rgb_out.ndim == 3 and rgb_out.shape[2] == 3 # HxWx3

        # basic_arc_env.render_mode = "human" # Hard to test non-interactively
        # human_out = basic_arc_env.render() # Prints to console
        # assert human_out is None

    finally:
        basic_arc_env.render_mode = original_mode # Restore

def test_get_challenge_info(basic_arc_env: ARCEnv):
    assert basic_arc_env is not None
    basic_arc_env.set_task("test_task_001")
    basic_arc_env.reset() # Task info is more complete after reset

    info = basic_arc_env.get_challenge_info()
    assert isinstance(info, dict)
    assert info.get("task_id") == "test_task_001"
    assert "num_train_pairs" in info
    assert "num_test_inputs" in info


@pytest.mark.slow # Example of a slow test marker
def test_arc_env_max_steps_truncation(basic_arc_env: ARCEnv):
    assert basic_arc_env is not None
    # Ensure max_steps is set for this test (dummy_env_config has max_steps=20)
    max_s = basic_arc_env.env_config.max_steps
    assert max_s is not None and max_s > 0

    basic_arc_env.set_task("test_task_001") # A task that is not trivially solved by random actions
    basic_arc_env.reset()

    terminated = False
    truncated = False
    for _ in range(max_s + 5): # Step more than max_steps
        action = basic_arc_env.action_space.sample()
        _, _, term, trunc, _ = basic_arc_env.step(action)
        if term: terminated = True; break
        if trunc: truncated = True; break

    assert not terminated, "Task should not have terminated with random actions within max_steps for this test."
    assert truncated is True, f"Environment should have truncated after {max_s} steps."
    assert basic_arc_env.num_steps_taken >= max_s


def test_arc_env_no_task_set_errors(dummy_env_config, dummy_action_space_config, dummy_operation_registry):
    """Test that errors are raised if methods are called before a task is set."""
    from arc_env.environments.arc_env import ARCEnv
    # Create env without initial_task_id and without a data_loader that can auto-load
    # This requires a data_loader that returns empty list of tasks, or pass None and handle.
    # For this test, let's assume data_loader is None and no default path, forcing error earlier.

    class EmptyLoader: # Mock loader
        def list_available_tasks(self): return []
        def load_task(self, task_id): raise TaskNotSetError(f"Task {task_id} not found in EmptyLoader")

    env_no_task = ARCEnv(
        env_config=dummy_env_config,
        action_space_config=dummy_action_space_config,
        data_loader=EmptyLoader(), # type: ignore
        operation_registry=dummy_operation_registry
    )

    with pytest.raises(TaskNotSetError):
        env_no_task.reset() # reset tries to load first task if none set, EmptyLoader has none.

    # If we manage to create an env where current_task_data is None after init:
    # (This is hard if constructor always tries to load a task or errors out)
    # A way is to have a loader that loads an invalid task initially.
    # For now, assume TaskNotSetError from reset covers the main cases.
    # If reset *succeeded* but somehow current_task_data was still None:
    # env_no_task.current_task_data = None # Manually force state for test (not good practice)
    # env_no_task.target_grid = None
    # with pytest.raises(TaskNotSetError):
    #     env_no_task.step(env_no_task.action_space.sample())


# Add more tests:
# - Specific action effects (e.g., a fill operation correctly fills selected area).
# - Correct reward calculation for various scenarios (step penalty, no-change, completion).
# - Observation content verification for different tasks.
# - Handling of different test_input_idx within a task.
# - Robustness to malformed task data (if not caught by loader/validator).
# - Interaction with selection mask.Tool output for `create_file_with_block`:
