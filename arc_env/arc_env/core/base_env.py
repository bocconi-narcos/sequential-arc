from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, Union

import gymnasium as gym
import numpy as np

class BaseARCEnvironment(gym.Env, ABC):
    """Abstract base class for ARC-style environments.

    This class defines the fundamental interface that all ARC environments
    within this package should adhere to, ensuring compatibility with
    Gymnasium's standards while also providing ARC-specific methods.
    """

    metadata: Dict[str, Any] = {"render_modes": [], "render_fps": 4} # Example, adjust as needed

    def __init__(self) -> None:
        super().__init__()
        # Define action_space and observation_space in concrete implementations
        # self.action_space: gym.Space
        # self.observation_space: gym.Space

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None, # Added options to match gym 0.26+
    ) -> Tuple[Any, Dict[str, Any]]: # Changed np.ndarray to Any for observation type flexibility
        """
        Reset the environment to its initial state.

        Args:
            seed: An optional seed for the random number generator.
            options: Optional dictionary of environment-specific options.

        Returns:
            A tuple containing the initial observation and an info dictionary.
        """
        # gymnasium.Env.reset() handles seeding of self.np_random
        super().reset(seed=seed, options=options)
        # Concrete implementations must then return obs, info
        pass

    @abstractmethod
    def step(
        self, action: Union[int, Dict[str, int]] # Action type can be more specific in implementations
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]: # Changed np.ndarray to Any
        """
        Execute one time step within the environment.

        Args:
            action: An action provided by the agent.

        Returns:
            A tuple containing:
                - observation: The agent's observation of the current environment.
                - reward: The amount of reward returned after previous action.
                - terminated: Whether the episode has ended (e.g., task solved).
                - truncated: Whether the episode has been truncated (e.g., time limit).
                - info: A dictionary containing auxiliary diagnostic information.
        """
        pass

    @abstractmethod
    def render(self) -> Union[np.ndarray, None]: # Adjusted render signature for gym 0.26+
        """
        Render the environment.

        The set of supported modes varies per environment. (Unsupported modes render nothing.)
        Usually human readable for human consumption and rgb_array for agent consumption.

        Returns:
            A numpy array representing the rendered frame if mode is 'rgb_array',
            otherwise None (if mode is 'human' and handled externally) or raises an error.
            Specific behavior depends on implementation and selected render_mode.
        """
        pass

    # render_mode is now typically a property or passed to __init__ in gym 0.26+
    # def render(self, mode: str = "human") -> Optional[np.ndarray]:
    # This was the old signature. The new one is simply render(self).
    # The mode is usually set during __init__ or as a property.
    # For flexibility, individual environments will manage their render_mode.

    @abstractmethod
    def get_challenge_info(self) -> Dict[str, Any]:
        """
        Get information about the current ARC challenge.

        This method should return a dictionary containing relevant details
        about the task, such as task ID, description, or specific constraints.

        Returns:
            A dictionary with challenge-specific information.
        """
        pass

    def close(self) -> None:
        """
        Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        super().close()
        pass

# Example of how observation_space and action_space would be defined in a subclass:
# class My конкреEnv(BaseARCEnvironment):
#     def __init__(self):
#         super().__init__()
#         self.observation_space = gym.spaces.Box(low=0, high=9, shape=(30, 30), dtype=np.uint8)
#         self.action_space = gym.spaces.Discrete(10) # Example
#
#     def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
#         super().reset(seed=seed)
#         # ... reset logic ...
#         return np.zeros((30,30), dtype=np.uint8), {}
#
#     def step(self, action):
#         # ... step logic ...
#         return np.zeros((30,30), dtype=np.uint8), 0.0, False, False, {}
#
#     def render(self):
#         # ... render logic ...
#         if self.render_mode == "rgb_array":
#             return np.zeros((30,30,3), dtype=np.uint8) # Example
#         return None
#
#     def get_challenge_info(self):
#         return {"id": "example_task"}
