"""
custom_solver.py

Demonstrates how to define a custom solver and use it with ARCEnv.
This example creates a simple heuristic solver.
"""
import numpy as np
from typing import Any, Dict, Optional, List
from pathlib import Path
import shutil # For cleanup

try:
    from arc_env.environments.arc_env import ARCEnv
    from arc_env.config.environment import EnvironmentConfig
    from arc_env.config.solver import SolverConfig # For custom solver config if needed
    from arc_env.solvers.base.base_solver import BaseSolver
    from arc_env.solvers.base.solver_registry import SolverRegistry
    from arc_env.spaces.action_spaces import ARCActionSpace # For type hints
    from arc_env.spaces.observation_spaces import ARCStructuredObservation # For type hints
except ImportError as e:
    print(f"ImportError: {e}. Ensure 'arc_env' is installed or in PYTHONPATH.")
    exit(1)

# --- Define a Custom Solver ---
class SimpleCopyInputSolver(BaseSolver):
    """
    A very simple custom solver that attempts to "solve" an ARC task
    by submitting the initial test input grid as its final solution.
    It doesn't take any real actions to modify the grid.
    This is for demonstration of the solver structure, not a useful solver.

    For a solver that takes actions, it would need to interact with the
    environment's action space. This example is simplified.
    """
    def __init__(self, solver_config: Optional[SolverConfig] = None, **kwargs):
        # If this solver had specific config, it would be a CustomSolverConfig(SolverConfig)
        # For now, use base SolverConfig.
        base_config = solver_config if solver_config else SolverConfig(solver_type="simple_copy_input")
        super().__init__(solver_config=base_config)
        self.action_space_ref: Optional[ARCActionSpace] = kwargs.get("action_space")
        print("SimpleCopyInputSolver initialized.")

    def predict_action(
        self,
        observation: ARCStructuredObservation,
        env_info: Optional[Dict[str, Any]] = None
    ) -> Any: # Returns an action compatible with env.action_space
        """
        This solver's strategy is to do nothing until the last step, then effectively
        "submit" the original input. Since it must return actions, it will return
        a "no-op" like action (e.g., first action of each category, or a specific
        no-op if defined in the action space).
        The actual "submission" is implicit if the environment terminates when
        current_grid matches target_grid. This solver doesn't actively make current_grid match.

        A true "submit current grid" action would be part of the DSL/action space.
        This placeholder will just take a minimal action.
        """
        # task_id = env_info.get("task_id", "Unknown") if env_info else "Unknown"
        # print(f"SimpleCopyInputSolver: 'Predicting' for task {task_id} (will return minimal action).")

        if self.action_space_ref:
            # Return a minimal action (e.g., first op from each category if factorized)
            if self.action_space_ref.config.mode == "factorized":
                # Construct action with index 0 for all available categories in the space
                # This assumes categories are in action_space_ref._category_keys
                # and underlying_gym_space.spaces reflects these.
                action = {}
                gym_dict_space = self.action_space_ref.underlying_gym_space
                if hasattr(gym_dict_space, 'spaces') and isinstance(gym_dict_space.spaces, dict):
                    for cat_key in gym_dict_space.spaces.keys():
                        action[cat_key] = 0 # First operation in this category
                    return action
            # For joint or other modes, or if factorized keys are complex:
            return self.action_space_ref.sample() # Fallback to random if construction is hard

        return 0 # Absolute fallback if no action space reference

    # This solver doesn't learn or have complex state, so other methods are simple.
    def reset_for_new_task(self) -> None:
        super().reset_for_new_task()
        # print(f"SimpleCopyInputSolver: Reset for task {self.current_task_id}")
        pass

    def episode_reset(self) -> None:
        super().episode_reset()
        # print(f"SimpleCopyInputSolver: Episode reset for task {self.current_task_id}")
        pass


def run_custom_solver_example(task_data_path: Path, num_episodes: int = 1, steps_per_episode: int = 3):
    print("\n--- ARC Environment: Custom Solver Example ---")

    # 1. Environment Setup (similar to basic_usage.py)
    env_cfg = EnvironmentConfig(canvas_size=10, max_steps=steps_per_episode + 2, data_path=task_data_path)
    # env_cfg.render_mode = "ansi"
    try:
        env = ARCEnv(env_config=env_cfg)
    except Exception as e:
        print(f"Error initializing ARCEnv for custom solver example: {e}")
        return

    # 2. Solver Registration (Optional, but good practice if using registry)
    solver_reg = SolverRegistry()
    solver_reg.register_solver("copy_input_solver", SimpleCopyInputSolver)

    # 3. Initialize Custom Solver
    # Pass the environment's action space to the solver if it needs it
    custom_solver_cfg = SolverConfig(solver_id="my_copy_solver_instance")
    try:
        my_solver = solver_reg.create_solver(
            "copy_input_solver",
            solver_config=custom_solver_cfg,
            action_space=env.action_space # Pass env's action space
        )
    except Exception as e:
        print(f"Error creating custom solver: {e}")
        env.close(); return

    # 4. Run with the custom solver
    task_to_load = "example_task_01" # From basic_usage helper
    try:
        env.set_task(task_to_load)
    except Exception as e:
        print(f"Error setting task '{task_to_load}' for custom solver example: {e}")
        env.close(); return

    for episode_num in range(num_episodes):
        print(f"\n--- Custom Solver: Episode {episode_num + 1} on Task '{env.current_task_id}' ---")
        observation, info = env.reset()

        # Update solver context for the task/episode
        my_solver.set_current_task_context(env.current_task_id or "unknown", env.get_challenge_info())
        my_solver.episode_reset()

        total_reward_episode = 0.0
        terminated = False
        truncated = False

        for step_num in range(steps_per_episode):
            if terminated or truncated: break

            print(f"\nStep {step_num + 1}:")

            # Solver predicts an action
            action = my_solver.predict_action(observation, info)
            # print(f"  Solver predicted action (raw): {action}")

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward_episode += reward

            print(f"  Env new state: Reward={reward:.2f}, Term={terminated}, Trunc={truncated}")
            # print(f"  Grid changed: {info.get('grid_changed')}")
            # env.render()

            # The SimpleCopyInputSolver strategy is passive. It won't solve most tasks.
            # A real solver would return actions to modify env.current_grid towards env.target_grid.
            # If, by some chance, the initial input grid IS the solution for a test case,
            # and the "no-op" actions from solver lead to no change, then terminated might be true.
            # For "example_task_01": input [[2,2],[2,2]], output [[3,3],[3,3]]. This solver won't solve it.

        print(f"\nEpisode {episode_num + 1} (Custom Solver) finished. Total reward: {total_reward_episode:.2f}")
        if terminated: print("  Task SOLVED by custom solver!")
        elif truncated: print("  Episode TRUNCATED for custom solver.")
        else: print("  Episode ended (max steps for example loop).")


    env.close()
    print("\n--- Custom Solver Example Finished ---")


if __name__ == "__main__":
    # Re-use helper from basic_usage to create dummy tasks
    from basic_usage import create_dummy_tasks_for_example # Assumes basic_usage.py is in same dir or PYTHONPATH

    temp_dir_for_custom_solver_ex = Path("_temp_custom_solver_data")
    example_tasks_path_custom = create_dummy_tasks_for_example(temp_dir_for_custom_solver_ex)

    try:
        run_custom_solver_example(task_data_path=example_tasks_path_custom, num_episodes=1, steps_per_episode=2)
    finally:
        if temp_dir_for_custom_solver_ex.exists():
            shutil.rmtree(temp_dir_for_custom_solver_ex)
            print(f"\nCleaned up temporary data for custom solver example: {temp_dir_for_custom_solver_ex}")
