from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, TypeVar, Generic, Optional
import gymnasium as gym
import numpy as np

# Generic type for the structured observation representation
StructuredObservationType = TypeVar('StructuredObservationType')

class BaseObservationSpace(gym.Space[Any], ABC, Generic[StructuredObservationType]):
    """
    Abstract base class for ARC observation spaces.

    This class extends Gymnasium's Space to provide a common interface
    for ARC-specific observation space functionalities. This might include
    methods to interpret or transform raw observations into more structured
    formats, or to provide metadata about the observation components.

    Concrete subclasses MUST define the `underlying_gym_space` property
    and ensure it returns a valid `gym.Space` instance after initialization.
    """

    def __init__(self) -> None:
        """
        Initializes the BaseObservationSpace.
        The actual shape, dtype will be determined by the `underlying_gym_space`
        that the concrete subclass will define and expose via overridden properties.
        """
        super().__init__() # Calls gym.Space.__init__(self, shape=None, dtype=None)

    @property
    @abstractmethod
    def underlying_gym_space(self) -> gym.Space:
        """
        Provides access to the actual Gymnasium space instance that this
        BaseObservationSpace wraps or defines. This is the space that
        the environment's `reset` and `step` methods will return observations from.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def structure_observation(self, observation: Any) -> StructuredObservationType:
        """
        Transforms a raw observation (typically a NumPy array or dict of arrays
        from the underlying gym.Space) into a more structured or domain-specific
        representation.

        Args:
            observation: The raw observation, compatible with the
                         `underlying_gym_space`.

        Returns:
            The structured observation.
        """
        pass

    # Delegate standard gym.Space methods to the underlying_gym_space.
    # Subclasses must ensure self.underlying_gym_space is properly initialized.

    def sample(self, mask: Optional[Any] = None) -> Any:
        """Return a sample from the underlying space."""
        return self.underlying_gym_space.sample(mask=mask)

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return self.underlying_gym_space.contains(x)

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        try:
            ugs_repr = repr(self.underlying_gym_space)
        except AttributeError:
            ugs_repr = "Underlying space not yet properly defined by subclass"
        return f"{self.__class__.__name__}({ugs_repr})"

    def __eq__(self, other: Any) -> bool:
        """Check if this space is equal to another."""
        if not isinstance(other, BaseObservationSpace):
            return False
        try:
            return self.underlying_gym_space == other.underlying_gym_space
        except AttributeError:
            return False

    def to_jsonable(self, sample_n: list[Any]) -> list[Any]:
        """Converts a list of samples from this space to a JSON serializable format."""
        return self.underlying_gym_space.to_jsonable(sample_n)

    def from_jsonable(self, sample_n: list[Any]) -> list[Any]:
        """Converts a list of JSON serializable samples back to this space's format."""
        return self.underlying_gym_space.from_jsonable(sample_n)

    # Override shape and dtype properties to delegate to the underlying_gym_space
    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        try:
            return self.underlying_gym_space.shape
        except AttributeError:
            return None

    @property
    def dtype(self) -> Optional[np.dtype]: # gymnasium.Space.dtype is Optional[np.dtype]
        try:
            return self.underlying_gym_space.dtype
        except AttributeError:
            return None


# Example of a concrete implementation (kept commented out):
# import gymnasium.spaces as spaces # Ensure this is imported if example is used
# from typing import TypedDict # Ensure this is imported if example is used

# class MyStructuredObs(TypedDict):
#     grid: np.ndarray
#     metadata: Dict[str, Any]

# class MySpecificObservationSpace(BaseObservationSpace[MyStructuredObs]):
#     def __init__(self, grid_shape: Tuple[int, ...]):
#         super().__init__()
#         self._concrete_gym_space_instance = spaces.Dict({
#             "grid": spaces.Box(low=0, high=9, shape=grid_shape, dtype=np.uint8),
#             "metadata": spaces.Dict({
#                 "task_id": spaces.Text(min_length=1, max_length=100)
#             })
#         })

#     @property
#     def underlying_gym_space(self) -> gym.Space: # type: ignore
#         if not hasattr(self, '_concrete_gym_space_instance'):
#              raise AttributeError("'_concrete_gym_space_instance' was not initialized.")
#         return self._concrete_gym_space_instance # type: ignore

#     def structure_observation(self, observation: Dict[str, Any]) -> MyStructuredObs:
#         if not self.underlying_gym_space.contains(observation):
#              raise ValueError("Observation is not compatible with the underlying space.")
#         # This cast assumes the keys and types match MyStructuredObs.
#         # A more robust implementation might perform explicit key checking and type conversion if needed.
#         return cast(MyStructuredObs, observation)
