from typing import Protocol, Any, Dict, Tuple, Union, Optional, runtime_checkable
import numpy as np
import gymnasium as gym

# Note: @runtime_checkable allows isinstance() checks, but has a performance cost.
# Use judiciously if runtime type checking of protocols is essential.

@runtime_checkable
class EnvironmentProtocol(Protocol):
    """
    Protocol defining the expected interface for an ARC environment.
    This aligns with BaseARCEnvironment and Gymnasium's Env.
    """
    action_space: gym.Space
    observation_space: gym.Space
    reward_range: Tuple[float, float]
    metadata: Dict[str, Any]
    render_mode: Optional[str]

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        ...

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        ...

    def render(self) -> Union[np.ndarray, None]:
        ...

    def close(self) -> None:
        ...

    def get_challenge_info(self) -> Dict[str, Any]:
        """Gets information about the current ARC challenge."""
        ...

@runtime_checkable
class ActionSpaceProtocol(Protocol):
    """
    Protocol for ARC action spaces, emphasizing the decode method.
    """
    underlying_gym_space: gym.Space

    def sample(self, mask: Optional[Any] = None) -> Any:
        ...

    def contains(self, x: Any) -> bool:
        ...

    def decode(self, action: Any) -> Any: # Return type can be generic in actual BaseActionSpace
        """Decodes a raw action into a structured/interpretable format."""
        ...

@runtime_checkable
class ObservationSpaceProtocol(Protocol):
    """
    Protocol for ARC observation spaces, emphasizing structuring observations.
    """
    underlying_gym_space: gym.Space
    shape: Optional[Tuple[int, ...]]
    dtype: Any


    def sample(self, mask: Optional[Any] = None) -> Any:
        ...

    def contains(self, x: Any) -> bool:
        ...

    def structure_observation(self, observation: Any) -> Any: # Return type can be generic
        """Transforms a raw observation into a structured format."""
        ...

@runtime_checkable
class SolverProtocol(Protocol):
    """
    Protocol defining the basic interface for an ARC solver.
    """
    def solve(self, observation: Any, env_info: Dict[str, Any]) -> Any: # ActionType
        """
        Given an observation and environment info, predict an action or solution.
        The exact types for observation and return action will depend on the
        specific environment and action space.
        """
        ...

    def reset(self) -> None:
        """Resets any internal state of the solver."""
        ...

@runtime_checkable
class ARCTaskDataProtocol(Protocol): # Renamed from ARCTaskData in base_loader.py for clarity here
    """Protocol for the structure of a single loaded ARC task."""
    train: List[Dict[str, np.ndarray]]
    test: List[Dict[str, np.ndarray]]
    task_id: Optional[str]

@runtime_checkable
class DataLoaderProtocol(Protocol):
    """
    Protocol for data loading components, aligning with BaseDataLoaderABC.
    """
    def load_task(self, task_identifier: Any) -> ARCTaskDataProtocol:
        """Loads a single ARC task."""
        ...

    def list_available_tasks(self) -> List[Any]:
        """Lists all available task identifiers."""
        ...

    # Optional methods from BaseDataLoaderABC could also be here if strictly needed by protocol users
    # def get_all_tasks(self) -> Dict[Any, ARCTaskDataProtocol]: ...
    # def validate_task_data(self, task_data: ARCTaskDataProtocol, task_id_for_error: Any) -> bool: ...


@runtime_checkable
class DataProcessorProtocol(Protocol):
    """
    Protocol for data processing components.
    """
    def process(self, data: Any) -> Any: # Input and output types depend on the processor
        """Processes the given data."""
        ...

@runtime_checkable
class OperationProtocol(Protocol):
    """
    Protocol for DSL operations.
    """
    def __call__(self, grid: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Applies the operation to a grid."""
        ...

    def to_string(self) -> str:
        """Returns a string representation of the operation."""
        ...

    # Could also include methods for parameter spaces, applicability checks, etc.


# Add more protocols as needed for other components like Loggers, Configurators, etc.

# Example usage (for illustration, not part of the file content itself):
#
# from .base_env import BaseARCEnvironment
#
# def process_env(env: EnvironmentProtocol) -> None:
#     obs, info = env.reset()
#     action = env.action_space.sample()
#     next_obs, reward, terminated, truncated, info = env.step(action)
#     challenge_info = env.get_challenge_info()
#     print(f"Challenge: {challenge_info.get('id', 'Unknown')}, Reward: {reward}")
#
# class MyEnv(BaseARCEnvironment):
#     # ... implementation ...
#     pass
#
# if __name__ == '__main__':
#    my_env_instance = MyEnv() # Assume MyEnv correctly implements the methods
#    if isinstance(my_env_instance, EnvironmentProtocol): # Requires @runtime_checkable
#        print("my_env_instance is compatible with EnvironmentProtocol")
#        process_env(my_env_instance)
#    else:
#        print("my_env_instance is NOT compatible with EnvironmentProtocol")
