from __future__ import annotations

from typing import Any, Dict, Tuple, Union, Optional, List, Callable, Protocol, TypeVar # Added Protocol, TypeVar
import gymnasium as gym
import numpy as np # Not directly used in this file's logic but common in env context

from .base import BaseARCWrapper
# from arc_env.data.loaders.base import ARCTaskData # Not directly used by this wrapper's logic beyond task_id list
from arc_env.exceptions import ConfigurationError, TaskNotSetError # Added TaskNotSetError

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

# Define a dummy protocol for type hinting `env.set_task` if needed for clarity.
# This is for type systems; runtime check is hasattr.
class SupportsSetTask(Protocol[ObsType, ActType]): # Made generic for env type
    def set_task(self, task_id: str) -> None: ...
    # Also inherit methods from gym.Env that CurriculumWrapper calls if strict typing needed for self.env
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]: ...
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]: ...
    # And get_challenge_info from BaseARCWrapper's expectation
    def get_challenge_info(self) -> Dict[str, Any]: ...


class CurriculumWrapper(BaseARCWrapper[ObsType, ActType, ObsType, ActType]):
    """
    A wrapper to manage curriculum learning for ARC environments.

    This wrapper can control the sequence of tasks presented to the agent,
    typically based on some measure of difficulty or a predefined order.
    It relies on the underlying environment to be able to load specific tasks.

    Note: This is a conceptual wrapper. The actual mechanism for changing tasks
    (e.g., via `env.set_task(task_id)` or specific `reset` options) needs to be
    supported by the wrapped ARC environment.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        task_sequence: List[str], # A list of task IDs in the desired curriculum order
        # Optional: A function to call when the curriculum advances or completes
        on_curriculum_step_fn: Optional[Callable[[str, int, int], None]] = None, # (new_task_id, current_idx, total_tasks)
        on_curriculum_complete_fn: Optional[Callable[[], None]] = None,
        loop_curriculum: bool = False # Whether to loop back to the start after finishing
    ):
        super().__init__(env)

        if not task_sequence:
            raise ConfigurationError("CurriculumWrapper requires a non-empty task_sequence (list of task IDs).")
        self.task_sequence = task_sequence
        self.num_tasks_in_curriculum = len(task_sequence)
        self.current_task_idx = -1 # Will be incremented to 0 on first reset

        self.on_curriculum_step_fn = on_curriculum_step_fn
        self.on_curriculum_complete_fn = on_curriculum_complete_fn
        self.loop_curriculum = loop_curriculum

        # Ensure the underlying environment has a way to set tasks.
        # This is a common pattern for ARC environments that support task switching.
        if not hasattr(self.env, 'set_task') or not callable(getattr(self.env, 'set_task')):
            raise NotImplementedError(
                "The wrapped environment for CurriculumWrapper must implement a 'set_task(task_id: str)' method."
            )

        # Initial task will be set by the first call to reset().

    def _set_next_task(self) -> Optional[str]:
        """
        Advances the curriculum to the next task and tells the environment to load it.
        Returns the new task_id if successful, None if curriculum completed and not looping.
        """
        self.current_task_idx += 1
        if self.current_task_idx >= self.num_tasks_in_curriculum:
            if self.loop_curriculum:
                self.current_task_idx = 0
                print("CurriculumWrapper: Looping curriculum back to start.")
            else:
                print("CurriculumWrapper: Curriculum completed.")
                if self.on_curriculum_complete_fn:
                    self.on_curriculum_complete_fn()
                return None # No more tasks

        new_task_id = self.task_sequence[self.current_task_idx]

        try:
            # Assuming self.env has a method like `set_task`
            # Use cast for type checker if self.env is gym.Env but we know it supports set_task
            from typing import cast
            cast(SupportsSetTask[ObsType, ActType], self.env).set_task(new_task_id)
            print(f"CurriculumWrapper: Advanced to task '{new_task_id}' ({self.current_task_idx + 1}/{self.num_tasks_in_curriculum}).")
            if self.on_curriculum_step_fn:
                self.on_curriculum_step_fn(new_task_id, self.current_task_idx, self.num_tasks_in_curriculum)
            return new_task_id
        except Exception as e:
            # If setting the task fails (e.g., task not found by underlying env's loader)
            # This is a critical error for the curriculum.
            raise RuntimeError(f"CurriculumWrapper: Failed to set task '{new_task_id}' in wrapped environment: {e}")


    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Resets the environment. If it's the start of the curriculum or an episode
        for the current task just finished, it advances to the next task in the sequence.
        """
        # The logic for advancing the curriculum is typically tied to when a task is "done"
        # (solved or max steps reached for that task).
        # A simple curriculum might advance on every reset.
        # A more complex one might only advance if the previous task was solved successfully
        # (e.g., based on info from the last step).

        # For this version, let's advance on every reset, which is common.
        # The first call to reset() will set the first task.

        new_task_id_set = self._set_next_task()
        if new_task_id_set is None: # Curriculum ended and not looping
            # What to do here? The environment cannot be reset to a new task.
            # Option 1: Raise an error.
            # Option 2: Keep resetting the last task (less 'curriculum'-like).
            # Option 3: Return a special observation indicating curriculum end.
            # For now, let's raise an error, or rely on the environment to handle
            # being reset without a new task if set_task wasn't called.
            # The `_set_next_task` already raises RuntimeError if set_task fails.
            # If it returns None, it means curriculum is done.
            # The underlying env.reset() will then run on its current (last) task.
            # This behavior might need to be more explicit.
            #
            # If curriculum is complete (and not looping), subsequent resets will just reset the *last* task.
            # This seems like reasonable default behavior if the user doesn't handle curriculum completion.
            print("CurriculumWrapper: Curriculum complete (not looping). Resetting current/last task.")
            pass # Allow env.reset() to proceed on its current task.

        # Now, call the underlying environment's reset.
        # The task should have been set by _set_next_task() via env.set_task().
        obs, info = self.env.reset(seed=seed, options=options)

        # Add curriculum info to the info dict
        info["curriculum_task_id"] = self.task_sequence[self.current_task_idx] if 0 <= self.current_task_idx < self.num_tasks_in_curriculum else None
        info["curriculum_task_index"] = self.current_task_idx
        info["curriculum_total_tasks"] = self.num_tasks_in_curriculum
        info["curriculum_is_complete"] = (new_task_id_set is None)

        return obs, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """
        Steps through the environment using the current task.
        The curriculum typically advances on `reset`, not on `step`.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add curriculum info to the step's info dict as well
        info["curriculum_task_id"] = self.task_sequence[self.current_task_idx] if 0 <= self.current_task_idx < self.num_tasks_in_curriculum else None
        info["curriculum_task_index"] = self.current_task_idx

        # If an episode ends (terminated or truncated), the next `reset` will handle curriculum progression.
        return obs, reward, terminated, truncated, info

# Define a dummy protocol for type hinting `env.set_task` if needed for clarity.
# This is just for illustration; structural typing (hasattr) is often used.
class SupportsSetTask(gym.Env[ObsType, ActType], Protocol):
    def set_task(self, task_id: str) -> None: ...


# Example Usage:
# if __name__ == "__main__":
#     # Needs a mock environment that implements `set_task(task_id)`
#     # and `get_challenge_info()`.
#     class MockTaskEnv(gym.Env):
#         def __init__(self, available_tasks: List[str]):
#             super().__init__()
#             self.available_tasks = available_tasks
#             self.current_task_id: Optional[str] = None
#             self.action_space = gym.spaces.Discrete(1)
#             self.observation_space = gym.spaces.Box(low=0, high=9, shape=(3,3), dtype=np.uint8)
#             self.set_task_called_with: Optional[str] = None

#         def set_task(self, task_id: str) -> None:
#             if task_id not in self.available_tasks:
#                 raise ValueError(f"MockTaskEnv: Task '{task_id}' not available.")
#             self.current_task_id = task_id
#             self.set_task_called_with = task_id # For testing
#             print(f"MockTaskEnv: Task set to '{self.current_task_id}'")

#         def reset(self, seed=None, options=None):
#             super().reset(seed=seed)
#             if self.current_task_id is None: # Should be set by wrapper before reset
#                 # Fallback: set to first available if not already set by curriculum
#                 if self.available_tasks: self.set_task(self.available_tasks[0])
#                 else: raise RuntimeError("MockTaskEnv: No task set and no available tasks.")

#             obs = self.observation_space.sample() # Dummy observation
#             # Simulate obs being related to task_id for clarity
#             if self.current_task_id == "task_easy": obs.fill(1)
#             elif self.current_task_id == "task_medium": obs.fill(5)
#             elif self.current_task_id == "task_hard": obs.fill(9)
#             return obs, {"current_task_in_env": self.current_task_id}

#         def step(self, action):
#             obs, info = self.reset() # Dummy step
#             reward = 0.0
#             terminated = random.choice([True, False]) # Randomly end episode
#             truncated = False
#             info["step_info_for_task"] = self.current_task_id
#             return obs, reward, terminated, truncated, info

#         def get_challenge_info(self) -> Dict[str, Any]: # For BaseARCWrapper
#             return {"id": self.current_task_id, "difficulty_mock": self.current_task_id}

#     available_mock_tasks = ["task_easy", "task_medium", "task_hard"]
#     base_env = MockTaskEnv(available_tasks=available_mock_tasks)

#     curriculum_task_order = ["task_easy", "task_medium", "task_hard"]

#     def on_step_callback(new_task_id, current_idx, total_tasks):
#         print(f"  Callback: Curriculum advanced to {new_task_id} ({current_idx+1}/{total_tasks})")
#     def on_complete_callback():
#         print(f"  Callback: Curriculum fully completed!")

#     curriculum_env = CurriculumWrapper(
#         base_env,
#         task_sequence=curriculum_task_order,
#         on_curriculum_step_fn=on_step_callback,
#         on_curriculum_complete_fn=on_complete_callback,
#         loop_curriculum=True # Test looping
#     )

#     print("--- Testing CurriculumWrapper ---")
#     for i in range(len(curriculum_task_order) + 2): # Go through curriculum and one extra to test loop/completion
#         print(f"\n--- Episode {i+1} ---")
#         obs, info = curriculum_env.reset()
#         print(f"Reset successful. Current curriculum task: {info.get('curriculum_task_id')}")
#         print(f"Observation (all same value for this mock task):\n{obs}")
#         assert info.get('curriculum_task_id') == base_env.set_task_called_with # Check env was told to set task

#         # Simulate a few steps in the episode
#         for step_num in range(2):
#             action = curriculum_env.action_space.sample()
#             obs, reward, terminated, truncated, step_info = curriculum_env.step(action)
#             print(f"  Step {step_num+1}: task='{step_info.get('curriculum_task_id')}', term={terminated}")
#             if terminated or truncated:
#                 print(f"  Episode for task '{step_info.get('curriculum_task_id')}' ended at step {step_num+1}.")
#                 break # Inner loop for steps

#     print("\n--- Test with non-looping curriculum ---")
#     curriculum_env_no_loop = CurriculumWrapper(
#         MockTaskEnv(available_mock_tasks), # Fresh env instance
#         task_sequence=curriculum_task_order,
#         loop_curriculum=False
#     )
#     for i in range(len(curriculum_task_order) + 1): # One more than tasks
#         print(f"\n--- NoLoop Episode {i+1} ---")
#         obs, info = curriculum_env_no_loop.reset()
#         is_complete = info.get("curriculum_is_complete", False)
#         task_id = info.get("curriculum_task_id")
#         print(f"Reset. Task: {task_id}, Curriculum Complete: {is_complete}")
#         if is_complete and i >= len(curriculum_task_order) : # After all tasks, task_id should be None or last task
#             assert task_id is None or task_id == curriculum_task_order[-1]
#             print(f"  (Confirmed: curriculum done, resetting last task: {curriculum_env_no_loop.env.current_task_id})") # type: ignore
#         elif task_id:
#              assert task_id == curriculum_task_order[i] # Check correct task progression
#         if i == len(curriculum_task_order): # This reset should indicate completion
#             assert is_complete is True

#     print("\nCurriculumWrapper tests finished.")
