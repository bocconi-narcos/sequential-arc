from __future__ import annotations

import gymnasium as gym
import numpy as np
from pathlib import Path
import shutil
import json # For dummy data creation helper

try:
    from arc_env.environments.arc_env import ARCEnv
    from arc_env.config.environment import EnvironmentConfig
    from arc_env.data.loaders.arc_loader import ARCFileLoader
    from arc_env.dsl.core.operation_registry import OperationRegistry
    from arc_env.dsl.presets.default import DefaultPresetLoader
except ImportError as e:
    print(f"ImportError: {e}. Ensure 'arc_env' is installed or in PYTHONPATH.")
    print("You might need to run `pip install .` or `pip install -e .` from the root of the project.")
    exit(1)


def create_dummy_tasks_for_example(base_dir: Path) -> Path:
    """Helper to create a temporary task directory for the example."""
    tasks_dir = base_dir / "example_arc_tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    task_content = {
        "example_task_01": {
            "train": [{"input": [[1,0],[0,1]], "output": [[0,1],[1,0]]}],
            "test": [{"input": [[2,2],[2,2]], "output": [[3,3],[3,3]]}]
        }
    }
    with open(tasks_dir / "example_task_01.json", "w") as f:
        json.dump(task_content["example_task_01"], f)
    return tasks_dir

def run_basic_usage_example(task_data_path: Path, num_episodes: int = 2, steps_per_episode: int = 5):
    """Runs the basic usage demonstration."""

    print("--- ARC Environment: Basic Usage Example ---")

    env_cfg = EnvironmentConfig(canvas_size=10, max_steps=(steps_per_episode + 5), data_path=task_data_path)
    # env_cfg.render_mode = "ansi"

    try:
        env = ARCEnv(env_config=env_cfg)
    except Exception as e:
        print(f"Error initializing ARCEnv: {e}")
        return

    print(f"Environment initialized. Action space: {env.action_space}")
    print(f"Observation space type: {type(env.observation_space.underlying_gym_space)}")

    task_to_load = "example_task_01"
    try:
        env.set_task(task_to_load)
        print(f"\nTask '{task_to_load}' set successfully.")
        challenge_info = env.get_challenge_info()
        print(f"Challenge Info: {challenge_info}")
    except Exception as e:
        print(f"Error setting task '{task_to_load}': {e}. Exiting example.")
        env.close()
        return

    for episode_num in range(num_episodes):
        print(f"\n--- Episode {episode_num + 1} ---")

        try:
            observation, info = env.reset()
        except Exception as e:
            print(f"Error resetting environment for episode {episode_num + 1}: {e}")
            continue

        print(f"Initial observation received for task '{info.get('task_id')}'.")

        total_reward_episode = 0.0
        terminated = False
        truncated = False

        for step_num in range(steps_per_episode):
            if terminated or truncated:
                break

            action = env.action_space.sample()

            print(f"\nStep {step_num + 1}:")

            try:
                observation, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                print(f"  Error during env.step(): {e}")
                break

            total_reward_episode += reward

            print(f"  Action (decoded): {info.get('action_decoded_str', 'N/A')}")
            print(f"  Reward: {reward:.2f}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")
            print(f"  Info: grid_changed={info.get('grid_changed')}, steps_taken={info.get('steps_taken_in_episode')}")

        print(f"\nEpisode {episode_num + 1} finished. Total reward: {total_reward_episode:.2f}")
        if terminated:
            print("  Task was SOLVED this episode!")
        elif truncated:
            print("  Episode TRUNCATED (max steps reached or other truncation).")

    env.close()
    print("\n--- Basic Usage Example Finished ---")


if __name__ == "__main__":
    temp_dir_for_example = Path("_temp_basic_usage_data")
    example_tasks_path = create_dummy_tasks_for_example(temp_dir_for_example)

    try:
        run_basic_usage_example(task_data_path=example_tasks_path, num_episodes=1, steps_per_episode=3)
    finally:
        if temp_dir_for_example.exists():
            shutil.rmtree(temp_dir_for_example)
            print(f"\nCleaned up temporary data directory: {temp_dir_for_example}")
