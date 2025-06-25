from __future__ import annotations

from typing import Any, Dict, Tuple, TypedDict, Optional, List # Added List for train_pairs in TypedDict
import gymnasium as gym
import numpy as np

from arc_env.core.base_observation_space import BaseObservationSpace # StructuredObservationType generic
from arc_env.config.environment import EnvironmentConfig # For default sizes, etc.

# Define the structure of a "structured" observation.
# This makes it easier to work with observation components.
class ARCStructuredObservation(TypedDict):
    """Typed dictionary for a structured ARC observation."""
    task_grid: np.ndarray       # The current state of the task's output grid being modified.
    train_pairs: list[Dict[str, np.ndarray]] # List of training examples, each a dict {"input": grid, "output": grid}
    test_input_grid: np.ndarray # The input grid for the current test task.
    # Optional elements:
    current_selection_mask: Optional[np.ndarray] # Boolean mask of the current selection
    # previous_action: Optional[Any] # Representation of the last action taken
    # current_step: Optional[int]
    # remaining_steps: Optional[int]
    # task_id: Optional[str]


class ARCObservationSpace(BaseObservationSpace[ARCStructuredObservation]):
    """
    Defines the observation space for ARC environments.

    The underlying gym.Space is typically a gym.spaces.Dict, containing:
    - 'task_grid': The main grid the agent interacts with and modifies.
    - 'train_pairs': A representation of the training examples for the current ARC task.
                     This is complex. Could be Box if fixed num examples & sizes, or Sequence.
                     For simplicity here, we might assume a fixed max number of pairs and
                     pad if fewer, or use a more flexible space type if available/needed.
                     A common approach is to have fixed max grid sizes for these.
    - 'test_input_grid': The input grid for the specific test case the agent needs to solve.

    Additional optional information can be part of the observation space or info dict.
    """

    def __init__(self, env_config: EnvironmentConfig, max_train_pairs: int = 3):
        """
        Args:
            env_config: Environment configuration, used for canvas sizes.
            max_train_pairs: The maximum number of training pairs an ARC task might have.
                             This is used to define a fixed-size observation space for train_pairs.
                             If a task has fewer, they will be padded.
        """
        super().__init__() # BaseObservationSpace init
        self.env_config = env_config
        self.max_train_pairs = max_train_pairs # ARC tasks usually have 2-5 train pairs.

        canvas_h, canvas_w = env_config.canvas_size, env_config.canvas_size # Assuming square

        # Define the shape and dtype for individual grid observations
        grid_obs_shape = (canvas_h, canvas_w)
        grid_dtype = np.uint8 # ARC colors 0-9

        # Define the observation space components
        obs_space_dict: Dict[str, gym.Space] = {
            "task_grid": gym.spaces.Box(
                low=0, high=9, shape=grid_obs_shape, dtype=grid_dtype
            ),
            "test_input_grid": gym.spaces.Box(
                low=0, high=9, shape=grid_obs_shape, dtype=grid_dtype
            )
        }

        # Handling train_pairs:
        # Each training pair has an 'input' and 'output' grid.
        # We use a Tuple space for a fixed number of pairs, each pair being a Dict of two Box spaces.
        pair_space = gym.spaces.Dict({
            "input": gym.spaces.Box(low=0, high=9, shape=grid_obs_shape, dtype=grid_dtype),
            "output": gym.spaces.Box(low=0, high=9, shape=grid_obs_shape, dtype=grid_dtype)
        })
        # Create a "padding" or "empty" pair representation for tasks with fewer than max_train_pairs.
        # This could be grids of a specific color (e.g., all -1 if Box low was -1, or specific shape).
        # For simplicity, we'll use the same space; actual padding logic is in env.
        obs_space_dict["train_pairs"] = gym.spaces.Tuple([pair_space] * self.max_train_pairs)


        # Optional components (can be added if always present in observation)
        # "current_selection_mask": gym.spaces.Box(
        #     low=0, high=1, shape=grid_obs_shape, dtype=bool # Or uint8
        # ),

        self._gym_space = gym.spaces.Dict(obs_space_dict)

        # BaseObservationSpace expects underlying_gym_space to be set.
        # Shape and dtype properties of BaseObservationSpace will delegate to this.
        # For a Dict space, shape is None.

    @property
    def underlying_gym_space(self) -> gym.Space:
        return self._gym_space

    def structure_observation(self, observation: Dict[str, Any]) -> ARCStructuredObservation:
        """
        Ensures the observation matches the ARCStructuredObservation TypedDict.
        This method primarily serves type hinting and validation, as the raw
        observation from gym.spaces.Dict should already match the structure.
        """
        if not self.underlying_gym_space.contains(observation):
            # This check is important. If it fails, the environment is producing
            # observations that don't conform to the defined space.
            raise ValueError("Raw observation is not contained within the underlying gym space.")

        # Perform runtime structural check if desired, or just cast for type system.
        # This assumes the keys in `observation` correctly map to ARCStructuredObservation.
        # For example, 'train_pairs' from gym.spaces.Tuple needs to be a list in TypedDict.

        structured_obs: ARCStructuredObservation = {
            "task_grid": observation["task_grid"],
            "train_pairs": list(observation["train_pairs"]), # Convert Tuple to list
            "test_input_grid": observation["test_input_grid"],
            # Optional fields:
            # "current_selection_mask": observation.get("current_selection_mask"),
        }
        return structured_obs

    def sample(self, mask: Optional[Any] = None) -> Dict[str, Any]:
        """Samples an observation from the space. Useful for testing."""
        return self.underlying_gym_space.sample(mask=mask)

    def __repr__(self) -> str:
        return f"ARCObservationSpace(underlying_space={self.underlying_gym_space})"

# Example Usage:
# if __name__ == "__main__":
#     from arc_env.config.environment import EnvironmentConfig
#     env_conf = EnvironmentConfig(canvas_size=10) # Example smaller canvas for display

#     arc_obs_space = ARCObservationSpace(env_config=env_conf, max_train_pairs=2)

#     print("ARCObservationSpace:")
#     print(arc_obs_space)
#     print("\nUnderlying Gym Dict Space:")
#     for key, space in arc_obs_space.underlying_gym_space.spaces.items():
#         print(f"  {key}: {space}")

#     print(f"\nShape of the Dict space: {arc_obs_space.shape}") # None for Dict
#     # print(f"Dtype of the Dict space: {arc_obs_space.dtype}") # Not well-defined for Dict

#     # Sample a raw observation
#     raw_sample_obs = arc_obs_space.sample()
#     print("\nSampled raw observation (dict from gym.spaces.Dict):")
#     # print("  task_grid:\n", raw_sample_obs["task_grid"])
#     # print("  train_pairs (tuple of dicts):\n", raw_sample_obs["train_pairs"]) # This will be a tuple
#     # print("  test_input_grid:\n", raw_sample_obs["test_input_grid"])

#     # Structure the observation
#     try:
#         structured_sample_obs = arc_obs_space.structure_observation(raw_sample_obs)
#         print("\nSampled structured observation (TypedDict):")
#         # print("  task_grid (shape):", structured_sample_obs["task_grid"].shape)
#         # print("  train_pairs (list of dicts, length):", len(structured_sample_obs["train_pairs"]))
#         # if structured_sample_obs["train_pairs"]:
#         #    print("    first pair input shape:", structured_sample_obs["train_pairs"][0]["input"].shape)
#         # print("  test_input_grid (shape):", structured_sample_obs["test_input_grid"].shape)

#         # Check if a valid observation is contained
#         is_contained = arc_obs_space.contains(raw_sample_obs)
#         print(f"\nIs the sampled observation contained in the space? {is_contained}") # True

#         # Example of an invalid observation
#         invalid_obs = raw_sample_obs.copy()
#         invalid_obs["task_grid"] = np.full((5,5), 15, dtype=np.uint8) # Wrong values and shape
#         is_invalid_contained = arc_obs_space.contains(invalid_obs)
#         print(f"Is the invalid observation contained in the space? {is_invalid_contained}") # False

#         # try:
#         #     arc_obs_space.structure_observation(invalid_obs) # Should raise ValueError
#         # except ValueError as e:
#         #     print(f"Error structuring invalid observation (expected): {e}")

#     except Exception as e:
#         print(f"An error occurred during example usage: {e}")
