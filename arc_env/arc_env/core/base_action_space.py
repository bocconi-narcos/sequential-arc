from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union, Tuple, Callable, TypeVar, Generic, Optional
import gymnasium as gym
import numpy as np # np.dtype is often used for gym space dtypes

# Generic type for the decoded action representation
DecodedActionType = TypeVar('DecodedActionType')

class BaseActionSpace(gym.Space[Any], ABC, Generic[DecodedActionType]):
    """
    Abstract base class for ARC action spaces.

    This class extends Gymnasium's Space to provide a common interface
    for ARC-specific action space functionalities, such as decoding
    actions into more complex, interpretable structures or operations.

    Concrete subclasses MUST define the `underlying_gym_space` property
    and ensure it returns a valid `gym.Space` instance after initialization.
    """

    def __init__(self) -> None:
        """
        Initializes the BaseActionSpace.
        The actual shape, dtype will be determined by the `underlying_gym_space`
        that the concrete subclass will define and expose via overridden properties.
        """
        super().__init__() # Calls gym.Space.__init__(self, shape=None, dtype=None)

    @property
    @abstractmethod
    def underlying_gym_space(self) -> gym.Space:
        """
        Provides access to the actual Gymnasium space instance that this
        BaseActionSpace wraps or defines. This is the space that agents
        will sample from and that the environment will expect actions from.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def decode(self, action: Any) -> DecodedActionType:
        """
        Decodes a raw action from the underlying gym.Space into a more
        structured or interpretable format specific to the ARC domain.

        Args:
            action: The raw action, typically an integer or a dictionary/tuple
                    of integers, compatible with the underlying `self.underlying_gym_space`.

        Returns:
            The decoded action in a domain-specific representation.
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
            # Access underlying_gym_space via property to ensure it's what subclass defined
            ugs_repr = repr(self.underlying_gym_space)
        except AttributeError:
            ugs_repr = "Underlying space not yet properly defined by subclass"
        return f"{self.__class__.__name__}({ugs_repr})"

    def __eq__(self, other: Any) -> bool:
        """Check if this space is equal to another."""
        if not isinstance(other, BaseActionSpace):
            return False
        try:
            # Comparison requires both underlying_gym_space properties to be valid.
            return self.underlying_gym_space == other.underlying_gym_space
        except AttributeError: # If one space's underlying_gym_space is not ready/defined
            return False

    # Override shape and dtype properties to delegate to the underlying_gym_space
    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        try:
            return self.underlying_gym_space.shape
        except AttributeError: # If underlying_gym_space isn't resolvable (subclass init error)
            # gym.Space.shape can return None if _shape is None.
            return None

    @property
    def dtype(self) -> Optional[np.dtype]: # gymnasium.Space.dtype is Optional[np.dtype]
        try:
            return self.underlying_gym_space.dtype
        except AttributeError:
            return None

    # seed method is deprecated in gym.Space and handled by Env.reset()
    # to_jsonable and from_jsonable are also delegated if underlying_gym_space implements them.
    def to_jsonable(self, sample_n: list[Any]) -> list[Any]:
        return self.underlying_gym_space.to_jsonable(sample_n)

    def from_jsonable(self, sample_n: list[Any]) -> list[Any]:
        return self.underlying_gym_space.from_jsonable(sample_n)


# Example of a concrete implementation (kept commented out):
# from typing import Callable # Ensure this is imported if example is used
# import gymnasium.spaces as spaces # Ensure this is imported if example is used

# class MySpecificActionSpace(BaseActionSpace[Tuple[Callable, ...]]):
#     def __init__(self, num_operations: int):
#         super().__init__() # Calls BaseActionSpace.__init__
#         self._operations_list = [lambda i=i: f"op_{i}" for i in range(num_operations)]
#         # This is the actual gym.Space that defines behavior for sample(), contains(), etc.
#         self._concrete_gym_space_instance = spaces.Discrete(num_operations)
#         # After _concrete_gym_space_instance is set, self.shape and self.dtype will work via delegation.

#     @property
#     def underlying_gym_space(self) -> gym.Space:
#         # This property must be implemented by all concrete subclasses.
#         # It should be available and valid after the subclass __init__ completes.
#         if not hasattr(self, '_concrete_gym_space_instance'): # Should always be true post-init
#              raise AttributeError("'_concrete_gym_space_instance' was not initialized.")
#         return self._concrete_gym_space_instance

#     def decode(self, action: int) -> Tuple[Callable, ...]: # Assuming action is int for Discrete
#         if not self.underlying_gym_space.contains(action): # `contains` will use self._concrete_gym_space_instance
#             raise ValueError(f"Action {action} is not valid for this space.")
#         return (self._operations_list[action],)
