import pytest
import numpy as np

from arc_env.environments.arc_env import ARCEnv # Main environment
# Fixtures used: basic_arc_env (provides a configured ARCEnv instance)

# Integration tests focus on the interaction between major components.
# Here, testing ARCEnv with its ARCActionSpace.

def test_env_action_space_compatibility(basic_arc_env: ARCEnv):
    """
    Tests that the environment's action space is an instance of ARCActionSpace
    and that sampling from it and stepping in the env works.
    """
    assert basic_arc_env is not None, "basic_arc_env fixture failed."
    from arc_env.spaces.action_spaces import ARCActionSpace
    assert isinstance(basic_arc_env.action_space, ARCActionSpace), \
        "Environment's action_space is not an ARCActionSpace instance."

    # Reset env to a known task state
    basic_arc_env.set_task("test_task_001") # Defined in conftest dummy_arc_tasks_dir
    obs, info = basic_arc_env.reset()

    # Sample an action from the ARCActionSpace
    # The type of raw_action depends on action_space_config.mode (factorized or joint)
    # basic_arc_env uses dummy_action_space_config, which defaults to "factorized".
    raw_action = basic_arc_env.action_space.sample()

    # Ensure the environment can accept this sampled action via step()
    try:
        next_obs, reward, terminated, truncated, next_info = basic_arc_env.step(raw_action)
    except Exception as e:
        pytest.fail(f"env.step() failed with a sampled action {raw_action}: {e}")

    # Basic checks on step output
    assert isinstance(next_obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(next_info, dict)
    assert "action_decoded_str" in next_info # ARCEnv adds this


def test_decoded_action_application(basic_arc_env: ARCEnv):
    """
    Tests if a decoded action (sequence of operations) can be applied and
    results in a changed grid state (or expected no-change).
    This is a deeper test of the env._apply_action_to_grid() logic.
    """
    assert basic_arc_env is not None
    basic_arc_env.set_task("test_task_001")
    initial_obs, _ = basic_arc_env.reset()
    initial_grid = np.copy(initial_obs["task_grid"]) # Agent's working grid

    # Construct a specific action that should cause a change.
    # This requires knowledge of available operations in the "default" preset
    # used by basic_arc_env's ARCActionSpace.
    # Example: SelectAll, then Fill with a color different from current dominant color.

    # Find "SelectAll" operation index
    select_all_op_idx = -1
    if "selection" in basic_arc_env.action_space._op_categories:
        for idx, op in enumerate(basic_arc_env.action_space._op_categories["selection"]):
            # This check is brittle as it depends on to_string() output.
            # A better way would be to check op class and params if known.
            if op.__class__.__name__ == "SelectAllOperation": # Assuming class name
                select_all_op_idx = idx
                break
    if select_all_op_idx == -1:
        pytest.skip("Could not find SelectAllOperation in 'selection' category for test.")

    # Find a "Fill" operation index (e.g., FillOperation with color 5)
    fill_color_5_op_idx = -1
    target_fill_color = 5 # A color likely different from initial grid's dominant color
    if "color" in basic_arc_env.action_space._op_categories:
        for idx, op in enumerate(basic_arc_env.action_space._op_categories["color"]):
            if op.__class__.__name__ == "FillOperation" and op.params.get("color") == target_fill_color:
                fill_color_5_op_idx = idx
                break

    if fill_color_5_op_idx == -1:
        # If specific Fill(5) not found, try to find *any* Fill and manually set its color for test action
        # This is harder as action is by index. For now, skip if specific Fill(5) isn't in preset.
        pytest.skip(f"Could not find FillOperation(color={target_fill_color}) in 'color' category for test.")

    # Assume action space is factorized (default for basic_arc_env)
    # And categories are "selection", "color", "transform" in that order for decoding.
    # This depends on ARCActionSpace._category_keys order.
    # Let's make the action dict based on actual category keys from the space.

    action_dict = {}
    found_sel = False
    found_col = False

    for cat_key in basic_arc_env.action_space._category_keys:
        if cat_key == "selection":
            action_dict[cat_key] = select_all_op_idx
            found_sel = True
        elif cat_key == "color":
            action_dict[cat_key] = fill_color_5_op_idx
            found_col = True
        else: # For other categories like "transform", pick first available op (index 0)
             if basic_arc_env.action_space._op_categories.get(cat_key): # If category has ops
                action_dict[cat_key] = 0

    if not (found_sel and found_col):
        pytest.skip("Required 'selection' or 'color' categories not found in action space for this test setup.")
    if not basic_arc_env.action_space.config.mode == "factorized":
        pytest.skip("This test logic assumes factorized action space mode.")

    # Perform the step with this constructed action
    next_obs, _, _, _, step_info = basic_arc_env.step(action_dict)
    changed_grid = step_info.get("grid_changed", False)

    assert changed_grid is True, "Grid should have changed after SelectAll and Fill(5)."

    # Verify the new grid state
    new_grid = next_obs["task_grid"]
    # After SelectAll then Fill(5), all pixels should be 5.
    # This assumes no other ops in the sequence undid the fill.
    # (e.g. if a transform op was index 0 and it was NoOp or identity)
    assert np.all(new_grid == target_fill_color), \
        f"Grid not filled with target color {target_fill_color}. Got:\n{new_grid[0:5,0:5]}"


def test_env_handles_max_steps(basic_arc_env: ARCEnv):
    """Tests if the environment correctly truncates after max_steps."""
    assert basic_arc_env is not None
    # Max steps is set in dummy_env_config (e.g., 20)
    max_s = basic_arc_env.env_config.max_steps
    assert max_s is not None and max_s > 0

    basic_arc_env.set_task("test_task_001") # A task not easily solved randomly
    basic_arc_env.reset()

    terminated = False
    truncated = False
    for i in range(max_s + 5): # Try to step beyond max_steps
        action = basic_arc_env.action_space.sample()
        _, _, term, trunc, info = basic_arc_env.step(action)

        if i < max_s -1 : # Before the last allowed step
            assert not trunc, f"Should not truncate at step {i+1} (max_steps={max_s})"
        elif i == max_s -1: # This is the max_steps'th step
            # Truncation should be True if not terminated
            if not term:
                assert trunc is True, f"Should truncate at step {i+1} if not terminated."
            # Store for final check
            terminated = term
            truncated = trunc
            break # Exit loop after max_steps is reached

    assert basic_arc_env.num_steps_taken == max_s
    assert truncated is True or (terminated is True and basic_arc_env.num_steps_taken <= max_s)
    # If it terminated exactly on the last step, truncated might also be true or false based on gym conventions.
    # Typically, if terminated, truncated is False. If time limit, truncated is True.
    # The important part is that it stops at max_steps if not solved.


# Add more integration tests:
# - Test with "joint" action space mode.
# - Test specific sequences of operations and their effects on the grid.
# - Test reward signals for different outcomes (solve, no-change, penalties).
# - Test interaction with different observation components (train_pairs, test_input_grid).
# - Test task switching and its effect on state and observations.
# - Test with MiniARCEnv or MultiTaskARCEnv variants if their interaction differs.
