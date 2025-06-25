from __future__ import annotations

import gymnasium as gym
from gymnasium.core import RewardWrapper, ObservationWrapper, ActionWrapper # For reference if creating such specific wrappers
from abc import ABC # ABC is used by BaseARCWrapper
from typing import Any, Dict, SupportsFloat, TypeVar, Generic # SupportsFloat for reward wrapper example

from arc_env.core.protocols import EnvironmentProtocol # For type checking the wrapped env

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


class BaseARCWrapper(gym.Wrapper[WrapperObsType, WrapperActType, ObsType, ActType], ABC):
    """
    Base wrapper for ARC environments.
    Ensures that the wrapped environment conforms to certain ARC-specific expectations,
    like having a `get_challenge_info` method.

    This class itself is still generic due to gym.Wrapper's own generics.
    Concrete wrappers will specify their types.
    If a wrapper doesn't change observation/action types, WrapperObsType=ObsType etc.
    """

    def __init__(self, env: gym.Env[ObsType, ActType]) -> None:
        """
        Initializes the wrapper.

        Args:
            env: The ARC environment to wrap. Must conform to EnvironmentProtocol.
        """
        super().__init__(env)
        self._validate_environment(env) # Validate before super().__init__ might be too early if env attributes are set up by gym.Wrapper
                                       # However, for protocol check, it might be fine.
                                       # Let's keep it after super().__init__() to be safe,
                                       # as `self.env` is set by it.

    def _validate_environment(self, env_to_check: gym.Env[ObsType, ActType]) -> None:
        """
        Validate that the wrapped environment is a compatible ARC environment.
        Specifically, it should have a `get_challenge_info` method.
        """
        # Check if it's an instance of our EnvironmentProtocol (if runtime_checkable)
        # or hasattr for specific methods.
        if not isinstance(env_to_check, EnvironmentProtocol): # Requires @runtime_checkable on Protocol
             # Fallback to hasattr if not runtime_checkable or for more specific checks
            if not hasattr(env_to_check, 'get_challenge_info') or not callable(getattr(env_to_check, 'get_challenge_info')):
                raise ValueError(
                    "The wrapped environment does not have a callable 'get_challenge_info' method, "
                    "which is expected for ARC environments."
                )
            # Could check for other ARC-specific attributes or methods here too.

    # Expose ARC-specific methods from the wrapped environment
    def get_challenge_info(self) -> Dict[str, Any]:
        """
        Get information about the current ARC challenge from the wrapped environment.
        """
        # self.env is the wrapped environment. We need to ensure it has this method.
        # The EnvironmentProtocol check should guarantee this if types are correct.
        if hasattr(self.env, 'get_challenge_info') and callable(getattr(self.env, 'get_challenge_info')):
            # The type hint for self.env is gym.Env, which doesn't guarantee get_challenge_info.
            # We cast to EnvironmentProtocol or Any to satisfy type checker if needed,
            # or rely on the runtime check in _validate_environment.
            return (self.env L ike EnvironmentProtocol).get_challenge_info() # type: ignore
        else:
            # This case should ideally be caught by _validate_environment.
            raise AttributeError("Wrapped environment is missing 'get_challenge_info' method despite initial validation.")

    # Other common wrapper methods (reset, step, render, close) are automatically
    # delegated by gym.Wrapper unless overridden.
    # For example, if a wrapper needs to modify observations, it would override `step` and `reset`.

    # Example of how a concrete wrapper might look (not part of this file):
    # class MySpecificARCWrapper(BaseARCWrapper[MyObsType, MyActType, OriginalObsType, OriginalActType]):
    #     def __init__(self, env: gym.Env[OriginalObsType, OriginalActType]):
    #         super().__init__(env)
    #         # Define new observation_space or action_space if they change
    #         # self.observation_space = ...
    #
    #     def observation(self, observation: OriginalObsType) -> MyObsType:
    #         # If it's an ObservationWrapper
    #         pass
    #
    #     def action(self, action: MyActType) -> OriginalActType:
    #         # If it's an ActionWrapper
    #         pass
    #
    #     def reward(self, reward: SupportsFloat) -> SupportsFloat:
    #         # If it's a RewardWrapper
    #         pass


# For convenience, we can also provide ARC-specific versions of common gym wrappers
# that inherit from BaseARCWrapper to ensure the get_challenge_info pass-through.

class ARCObservationWrapper(ObservationWrapper, BaseARCWrapper[WrapperObsType, ActType, ObsType, ActType], Generic[WrapperObsType, ActType, ObsType]):
    """ARC-specific ObservationWrapper."""
    # This class combines ObservationWrapper's observation processing with
    # BaseARCWrapper's ARC-specific validation and method exposure.
    # Due to MRO and generic complexities, gym.Wrapper should usually be the first parent
    # if we are not re-implementing its core logic.
    # However, ObservationWrapper itself inherits from Wrapper.
    # Let's try: class ARCObservationWrapper(BaseARCWrapper, ObservationWrapper):
    # The order of generics in BaseARCWrapper is [WrapperObsType, WrapperActType, ObsType, ActType]
    # ObservationWrapper is [ObsType, ActType, EnvObsType] -> [WrapperObsType, ActType, ObsType] if WrapperObsType is new obs type
    # This gets complicated. A simpler approach for concrete wrappers:
    # class MyWrapper(BaseARCWrapper[NewObs, Act, OldObs, Act]):
    #    def observation(self, obs:OldObs) -> NewObs: ...

    # Let's redefine more simply:
    # BaseARCWrapper is fine. Concrete wrappers will inherit from BaseARCWrapper
    # AND the specific gym.ObservationWrapper, gym.ActionWrapper, etc.
    # This composition is tricky with Python's MRO if both parents define similar methods (like __getattr__).
    # The gym.Wrapper is quite sophisticated.
    #
    # The provided skeleton for ActionMaskingWrapper directly inherits BaseARCWrapper.
    # This implies BaseARCWrapper is the primary parent and specific wrapper logic (obs/act/reward modification)
    # is implemented directly by overriding step/reset. This is a valid approach.
    #
    # If we wanted to make it easy to create ARC wrappers that also use the gym.ObservationWrapper pattern:
    # class ARCObservationWrapper(ObservationWrapper, BaseARCWrapper):
    #    # This would require careful handling of __init__ and super() calls.
    #    # gym.ObservationWrapper(env)
    #    # BaseARCWrapper(env) -> this would call super().__init__(env) twice if not careful.
    #    # gym.Wrapper is designed to be the primary one.
    #    pass # This is non-trivial to get right.

    # Sticking to the pattern: BaseARCWrapper is the main one.
    # If a wrapper IS an observation wrapper, it overrides observation() if it inherits from gym.ObservationWrapper,
    # or it manually processes obs in step() and reset() if it only inherits from BaseARCWrapper or gym.Wrapper.
    # The given ActionMaskingWrapper skeleton suggests the latter (manual override of step).
    pass # BaseARCWrapper is sufficient as a starting point. Concrete wrappers will show patterns.
