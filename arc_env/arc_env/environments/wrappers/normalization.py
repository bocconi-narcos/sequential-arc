from __future__ import annotations

from typing import Any, Dict, Tuple, Union, Optional, TypeVar, List # Added List
import gymnasium as gym
import numpy as np

from .base import BaseARCWrapper
# from arc_env.spaces.observation_spaces import ARCObservationSpace # If we need to check specific obs structure

ObsType = TypeVar("ObsType") # Original observation type from wrapped env
ActType = TypeVar("ActType") # Action type (usually not changed by obs norm wrapper)
NormObsType = TypeVar("NormObsType") # Normalized observation type
# For the Generic inheritance of ObservationWrapper
WrapperObsType = NormObsType # Alias for clarity in gym.ObservationWrapper generic signature
EnvObsType = ObsType        # Alias for clarity
EnvActType = ActType        # Alias for clarity

# For ARC, raw grid observations are typically uint8 in [0,9]. Normalization to [0,1] or [-1,1]
# might be useful for some neural network inputs. This wrapper would handle that.
# It needs to modify the observation_space as well.

class ObservationNormalizationWrapper(
    BaseARCWrapper[NormObsType, ActType, ObsType, ActType], # Wrapper generics
    gym.ObservationWrapper[NormObsType, ObsType, ActType]    # gym.ObservationWrapper generics
):
    """
    A wrapper to normalize observations from an ARC environment.
    This example assumes observations are dictionaries containing grid-like numpy arrays
    (e.g., 'task_grid', 'test_input_grid' from ARCObservationSpace) and normalizes
    their values from a known range (e.g., 0-9 for ARC colors) to a new range (e.g., 0.0-1.0).

    Note: This is a conceptual example. The actual structure of ObsType and NormObsType,
    and which parts of the observation to normalize, depend on the specific ARCObservationSpace.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        min_val: float = 0.0,
        max_val: float = 9.0,       # Original range of ARC colors
        norm_min: float = 0.0,
        norm_max: float = 1.0,    # Target normalized range
        keys_to_normalize: Optional[List[str]] = None # Specify which keys in Dict obs to normalize
                                                      # If None, attempts to normalize all ndarray fields.
        ):
        # We need to call both super().__init__ methods correctly.
        # gym.ObservationWrapper.__init__(self, env)
        # BaseARCWrapper.__init__(self, env)
        # Since ObservationWrapper is a gym.Wrapper, its __init__ calls gym.Wrapper.__init__
        # BaseARCWrapper also calls gym.Wrapper.__init__.
        # This MRO can be tricky. Standard way: call super of current class.
        super().__init__(env) # This calls gym.ObservationWrapper's init, which calls gym.Wrapper's init.
                              # BaseARCWrapper's validation will also run due to its position in MRO,
                              # but its own super().__init__(env) call might be redundant or skipped
                              # depending on how gym.ObservationWrapper calls super.
                              # Let's ensure BaseARCWrapper's specific logic (validation) runs.

        self._validate_environment(env) # Explicitly call BaseARCWrapper's validation.

        self.original_min = min_val
        self.original_max = max_val
        self.norm_min = norm_min
        self.norm_max = norm_max

        if self.original_max == self.original_min:
            raise ValueError("original_max and original_min cannot be equal for normalization.")

        self.keys_to_normalize = keys_to_normalize

        # Modify the observation space
        # This requires knowing the structure of the original observation space.
        # If it's a Dict space, we update the Box spaces within it.
        self.observation_space = self._transform_observation_space(env.observation_space)


    def _transform_observation_space(self, obs_space: gym.Space) -> gym.Space:
        """
        Transforms the original observation space to reflect normalization.
        If it's a Dict space, it iterates through its components.
        """
        if isinstance(obs_space, gym.spaces.Dict):
            new_spaces: Dict[str, gym.Space] = {}
            for key, space_component in obs_space.spaces.items():
                if isinstance(space_component, gym.spaces.Box):
                    if self.keys_to_normalize is None or key in self.keys_to_normalize:
                        # Assuming dtype becomes float after normalization
                        new_spaces[key] = gym.spaces.Box(
                            low=self.norm_min,
                            high=self.norm_max,
                            shape=space_component.shape,
                            dtype=np.float32 # Or appropriate float type
                        )
                    else: # Not normalizing this key
                        new_spaces[key] = space_component
                elif isinstance(space_component, gym.spaces.Tuple): # Handle Tuple of Boxes, e.g. train_pairs
                    new_tuple_components = []
                    for i, tuple_elem_space in enumerate(space_component.spaces):
                         # Recursively transform, or handle known structures like ARCObservationSpace's train_pairs
                         if isinstance(tuple_elem_space, gym.spaces.Dict) and \
                            all(isinstance(s, gym.spaces.Box) for s in tuple_elem_space.spaces.values()):
                             # This looks like a train_pair {"input": Box, "output": Box}
                             new_pair_dict_space = {}
                             for pair_key, pair_box_space in tuple_elem_space.spaces.items():
                                 # Check if "train_pairs.[input/output]" should be normalized based on keys_to_normalize
                                 # This requires a more complex key matching (e.g. "train_pairs.input")
                                 # For simplicity, assume all Box spaces within specified keys are normalized,
                                 # or all Box spaces if keys_to_normalize is None.
                                 # Here, let's assume if "train_pairs" is a key to normalize, its sub-Boxes are normalized.
                                 should_normalize_sub_key = (self.keys_to_normalize is None or
                                                             key in self.keys_to_normalize or
                                                             f"{key}.{pair_key}" in self.keys_to_normalize)

                                 if isinstance(pair_box_space, gym.spaces.Box) and should_normalize_sub_key:
                                     new_pair_dict_space[pair_key] = gym.spaces.Box(
                                        low=self.norm_min, high=self.norm_max,
                                        shape=pair_box_space.shape, dtype=np.float32
                                     )
                                 else:
                                     new_pair_dict_space[pair_key] = pair_box_space
                             new_tuple_components.append(gym.spaces.Dict(new_pair_dict_space))
                         else: # Non-dict-of-box element in tuple, keep as is or error
                             new_tuple_components.append(tuple_elem_space)
                    new_spaces[key] = gym.spaces.Tuple(new_tuple_components)
                else: # Non-Box component, keep as is
                    new_spaces[key] = space_component
            return gym.spaces.Dict(new_spaces) # type: ignore # Cast to NormObsType if it matches this structure
        elif isinstance(obs_space, gym.spaces.Box): # If the whole obs is a single Box
            return gym.spaces.Box(
                low=self.norm_min,
                high=self.norm_max,
                shape=obs_space.shape,
                dtype=np.float32
            ) # type: ignore
        else:
            # This wrapper might only be suitable for Dict or Box observation spaces.
            raise ValueError(f"ObservationNormalizationWrapper not implemented for space type: {type(obs_space)}")


    def observation(self, observation: ObsType) -> NormObsType:
        """
        Applies normalization to the observation.
        This method is called by gym.ObservationWrapper logic.
        """
        if isinstance(observation, dict): # Assuming ObsType is a Dict-like structure
            norm_obs = {}
            for key, value in observation.items():
                if isinstance(value, np.ndarray) and (self.keys_to_normalize is None or key in self.keys_to_normalize):
                    norm_obs[key] = self._normalize_array(value)
                elif isinstance(value, (list, tuple)) and (self.keys_to_normalize is None or key in self.keys_to_normalize):
                    # Handle list/tuple of dicts (like train_pairs)
                    norm_obs[key] = []
                    for item in value:
                        if isinstance(item, dict): # e.g. a train pair {"input": arr, "output": arr}
                            norm_item_dict = {}
                            for sub_key, sub_val_arr in item.items():
                                if isinstance(sub_val_arr, np.ndarray):
                                     # Check if "key.sub_key" should be normalized if keys_to_normalize is specific
                                     norm_item_dict[sub_key] = self._normalize_array(sub_val_arr)
                                else:
                                     norm_item_dict[sub_key] = sub_val_arr # Should not happen for train_pairs
                            norm_obs[key].append(norm_item_dict)
                        else: # Should not happen for train_pairs
                            norm_obs[key].append(item)
                else: # Non-array value or key not in keys_to_normalize
                    norm_obs[key] = value
            return norm_obs # type: ignore # Cast to NormObsType

        elif isinstance(observation, np.ndarray): # If observation is a single array
            return self._normalize_array(observation) # type: ignore
        else:
            raise TypeError(f"Observation type {type(observation)} not supported for normalization by this wrapper.")

    def _normalize_array(self, array: np.ndarray) -> np.ndarray:
        """Helper to normalize a single numpy array."""
        # Convert to float first to avoid issues with int types
        float_array = array.astype(np.float32)
        # Apply normalization formula: (val - orig_min) / (orig_max - orig_min) * (norm_range) + norm_min
        normalized = (float_array - self.original_min) / (self.original_max - self.original_min)
        normalized = normalized * (self.norm_max - self.norm_min) + self.norm_min
        return normalized.astype(np.float32) # Ensure output is float

# Example Usage:
# if __name__ == "__main__":
#     from arc_env.environments.arc_env import ARCEnv # Assuming this will exist
#     from arc_env.config.environment import EnvironmentConfig
#     from arc_env.spaces.observation_spaces import ARCObservationSpace # For sample obs structure

#     # This example needs a mock environment or a fully implemented ARCEnv
#     # Let's create a mock environment that has a compatible observation space
#     class MockARCEnv(gym.Env):
#         def __init__(self):
#             super().__init__()
#             self.env_config = EnvironmentConfig(canvas_size=5)
#             self.observation_space = ARCObservationSpace(self.env_config, max_train_pairs=1).underlying_gym_space
#             self.action_space = gym.spaces.Discrete(1) # Dummy action space
#         def reset(self, seed=None, options=None):
#             super().reset(seed=seed)
#             # Return a sample observation matching ARCObservationSpace structure
#             obs_sample = self.observation_space.sample()
#             # Ensure grids have values in 0-9 range for normalization test
#             obs_sample["task_grid"] = np.random.randint(0, 10, self.env_config.canvas_size**2).reshape(self.env_config.canvas_size, self.env_config.canvas_size)
#             obs_sample["test_input_grid"] = np.random.randint(0, 10, self.env_config.canvas_size**2).reshape(self.env_config.canvas_size, self.env_config.canvas_size)
#             # train_pairs is a tuple of dicts
#             obs_sample["train_pairs"][0]["input"] = np.random.randint(0, 10, self.env_config.canvas_size**2).reshape(self.env_config.canvas_size, self.env_config.canvas_size)
#             obs_sample["train_pairs"][0]["output"] = np.random.randint(0, 10, self.env_config.canvas_size**2).reshape(self.env_config.canvas_size, self.env_config.canvas_size)
#             return obs_sample, {"info": "mock_reset"}
#         def step(self, action):
#             obs, _ = self.reset() # For simplicity, step returns a new obs
#             return obs, 0.0, False, False, {"info": "mock_step"}
#         def render(self): pass
#         def close(self): pass
#         def get_challenge_info(self): return {"id": "mock_task"}

#     mock_env = MockARCEnv()
#     # Keys to normalize in the Dict observation (matching ARCObservationSpace)
#     norm_keys = ["task_grid", "test_input_grid", "train_pairs"] # train_pairs will be handled recursively

#     norm_wrapper = ObservationNormalizationWrapper(
#         mock_env,
#         min_val=0.0, max_val=9.0, # Original ARC color range
#         norm_min=0.0, norm_max=1.0, # Target range [0,1]
#         keys_to_normalize=norm_keys
#     )

#     print("Original Observation Space (Box component for 'task_grid'):")
#     print(mock_env.observation_space["task_grid"])
#     print("\nNormalized Observation Space (Box component for 'task_grid'):")
#     print(norm_wrapper.observation_space["task_grid"])
#     print("\nNormalized Observation Space (Tuple component for 'train_pairs'[0]['input']):")
#     print(norm_wrapper.observation_space["train_pairs"].spaces[0]["input"])


#     obs, info = norm_wrapper.reset()
#     print("\nSample Normalized Observation ('task_grid' component):")
#     print(obs["task_grid"])
#     print(f"  Min value: {np.min(obs['task_grid'])}, Max value: {np.max(obs['task_grid'])}")
#     print("  Dtype:", obs["task_grid"].dtype)

#     print("\nSample Normalized Observation ('train_pairs'[0]['input'] component):")
#     print(obs["train_pairs"][0]["input"])
#     print(f"  Min value: {np.min(obs['train_pairs'][0]['input'])}, Max value: {np.max(obs['train_pairs'][0]['input'])}")


#     # Test get_challenge_info through wrapper
#     print("\nChallenge info via wrapper:", norm_wrapper.get_challenge_info())
