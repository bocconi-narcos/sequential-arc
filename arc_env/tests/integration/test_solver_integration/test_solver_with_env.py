import pytest
import numpy as np

from arc_env.environments.arc_env import ARCEnv
from arc_env.solvers.base.base_solver import BaseSolver
from arc_env.solvers.implementations.heuristic.placeholder_heuristic_solver import PlaceholderHeuristicSolver # Example
from arc_env.config.solver import HeuristicSolverConfig
# Fixtures used: basic_arc_env

# Integration tests for a solver interacting with the ARCEnv.

@pytest.fixture
def placeholder_heuristic_solver(basic_arc_env: ARCEnv) -> PlaceholderHeuristicSolver:
    """Provides a PlaceholderHeuristicSolver instance configured for the basic_arc_env."""
    assert basic_arc_env is not None, "basic_arc_env fixture failed."
    config = HeuristicSolverConfig(search_depth=1) # Minimal config for placeholder
    # Placeholder solver needs action_space from env to sample actions
    return PlaceholderHeuristicSolver(solver_config=config, action_space=basic_arc_env.action_space)


def test_solver_predict_and_env_step(basic_arc_env: ARCEnv, placeholder_heuristic_solver: PlaceholderHeuristicSolver):
    """
    Tests the basic loop: solver predicts action, environment steps with it.
    """
    assert basic_arc_env is not None
    assert placeholder_heuristic_solver is not None

    task_id = "test_task_001"
    basic_arc_env.set_task(task_id)
    obs, info = basic_arc_env.reset()

    # Update solver's context about the current task
    challenge_info = basic_arc_env.get_challenge_info()
    placeholder_heuristic_solver.set_current_task_context(task_id, challenge_info)
    placeholder_heuristic_solver.episode_reset() # Reset for the new episode

    # Solver predicts an action
    try:
        action = placeholder_heuristic_solver.predict_action(obs, info)
    except Exception as e:
        pytest.fail(f"Solver failed to predict action: {e}")

    # Environment takes the action
    try:
        next_obs, reward, terminated, truncated, next_info = basic_arc_env.step(action)
    except Exception as e:
        pytest.fail(f"Environment failed to step with solver's action {action}: {e}")

    # Basic checks on outputs
    assert isinstance(next_obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(next_info, dict)


def test_solver_completes_a_task_episode(basic_arc_env: ARCEnv, placeholder_heuristic_solver: PlaceholderHeuristicSolver):
    """
    Simulates a solver running for a full episode on a task, until termination or truncation.
    """
    assert basic_arc_env is not None
    assert placeholder_heuristic_solver is not None

    task_id = "test_task_001" # A task from conftest dummy data
    basic_arc_env.set_task(task_id)
    obs, info = basic_arc_env.reset()

    challenge_info = basic_arc_env.get_challenge_info()
    placeholder_heuristic_solver.set_current_task_context(task_id, challenge_info)
    placeholder_heuristic_solver.episode_reset()

    max_episode_steps = basic_arc_env.env_config.max_steps or 50 # Use env's max_steps or a default
    if max_episode_steps is None: max_episode_steps = 50 # Ensure it's not None for test loop

    done = False
    total_reward = 0.0
    steps_taken = 0

    for step_count in range(max_episode_steps):
        steps_taken = step_count + 1
        action = placeholder_heuristic_solver.predict_action(obs, info)
        obs, reward, terminated, truncated, info = basic_arc_env.step(action)
        total_reward += reward

        if terminated or truncated:
            done = True
            print(f"Episode for task '{task_id}' finished after {steps_taken} steps. "
                  f"Terminated: {terminated}, Truncated: {truncated}, Total Reward: {total_reward:.2f}")
            break

    assert done, f"Episode should have finished (term or trunc) within {max_episode_steps} steps."
    # PlaceholderHeuristicSolver uses random actions, so solving is unlikely.
    # We mainly check that the loop runs and terminates/truncates as expected by env limits.
    if not terminated: # If not solved
        assert truncated is True, "If not terminated, should be truncated by max_steps."
        assert steps_taken == max_episode_steps, "If truncated, steps_taken should equal max_episode_steps."

# More integration tests could involve:
# - A "smarter" dummy solver that attempts a known sequence of actions to solve a specific dummy task.
# - Testing how different solver configurations (passed via SolverConfig) affect behavior with the env.
# - If solvers have learning/saving/loading, testing that flow in conjunction with env interactions.
# - Testing with different environment variants (MiniARCEnv, MultiTaskARCEnv) if solver interaction changes.
# - Using the BenchmarkRunner with a simple solver and dataset to test the full evaluation pipeline.
