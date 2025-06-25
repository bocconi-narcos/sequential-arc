from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Any, Dict, List, Union, Optional, Tuple # Added Tuple

# This module can contain utility functions related to Gymnasium spaces,
# especially for custom spaces or complex space transformations used in ARC.

def flatten_space(space: gym.Space) -> gym.Space:
    """
    Flattens a possibly nested space (like Dict or Tuple) into a single Box space.
    This is useful for some RL algorithms that require flat observation or action spaces.

    Note: This is a simplified example. `gym.spaces.utils.flatten_space`
    is the standard way to do this for built-in gym spaces. This function
    might be useful if custom flattening logic is needed for specific ARC structures.
    """
    # For most cases, gym's utility is preferred:
    # from gymnasium.spaces.utils import flatten_space as gym_flatten_space
    # return gym_flatten_space(space)

    # Example of custom handling (if gym's isn't sufficient or for learning):
    if isinstance(space, gym.spaces.Box):
        return space # Already flat in a sense (or can be flattened further if multi-dim)
    elif isinstance(space, gym.spaces.Discrete):
        # Convert Discrete to a one-hot encoded Box space
        return gym.spaces.Box(low=0, high=1, shape=(space.n,), dtype=np.float32) # Or int8
    elif isinstance(space, gym.spaces.Dict):
        # Recursively flatten components and concatenate them
        # This requires careful management of shapes and dtypes.
        # `gym.spaces.utils.flatten_space` handles this robustly.
        # This is a good place to use the standard utility.
        from gymnasium.spaces.utils import flatten_space as gym_flatten_space
        return gym_flatten_space(space)
    elif isinstance(space, gym.spaces.Tuple):
        from gymnasium.spaces.utils import flatten_space as gym_flatten_space
        return gym_flatten_space(space)
    else:
        raise NotImplementedError(f"Flattening not implemented for space type: {type(space)}")


def unflatten_action(flat_action: np.ndarray, original_space: gym.Space) -> Any:
    """
    Converts a flattened action back into the original nested action space structure.
    This is the inverse of `flatten_space`.

    Note: `gym.spaces.utils.unflatten` is the standard way.
    """
    # from gymnasium.spaces.utils import unflatten as gym_unflatten
    # return gym_unflatten(original_space, flat_action)

    # Using gym's utility is highly recommended here due to complexity.
    from gymnasium.spaces.utils import unflatten as gym_unflatten
    return gym_unflatten(original_space, flat_action)

def get_space_size(space: gym.Space) -> Union[int, float]:
    """
    Calculates the 'size' of a space.
    - For Discrete: number of actions.
    - For Box: product of shape dimensions (if bounded, could be volume, but usually number of elements).
    - For Dict/Tuple: sum or product of component sizes (depending on interpretation).
    Returns float('inf') for unbounded Box spaces.
    """
    if isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.Box):
        if not space.is_bounded():
            return float('inf')
        return int(np.prod(space.shape)) # Number of elements in the Box
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return int(np.prod(space.nvec)) # Total number of discrete combinations
    elif isinstance(space, gym.spaces.MultiBinary):
        return 2**space.n # Total number of binary combinations
    elif isinstance(space, gym.spaces.Tuple):
        # Could be sum or product depending on how tuple elements are combined.
        # If they are independent choices, product is more like "total combinations".
        # If it's a sequence, sum of element sizes might be more relevant.
        # Let's assume product for combination size.
        size = 1
        for s_comp in space.spaces:
            comp_size = get_space_size(s_comp)
            if comp_size == float('inf'): return float('inf')
            size *= comp_size
        return size
    elif isinstance(space, gym.spaces.Dict):
        # Similar to Tuple, interpretation matters.
        # Product of component sizes for total combinations.
        size = 1
        for s_comp in space.spaces.values():
            comp_size = get_space_size(s_comp)
            if comp_size == float('inf'): return float('inf')
            size *= comp_size
        return size
    else:
        raise NotImplementedError(f"Size calculation not implemented for space type: {type(space)}")


def create_empty_sample(space: gym.Space, fill_value: Optional[Any] = 0) -> Any:
    """
    Creates a sample that conforms to the space structure, but filled with a
    specific value (e.g., for padding or representing an empty/null state).

    Args:
        space: The gym.Space to create a sample for.
        fill_value: The value to fill elements with. Default is 0.
                    For Discrete, this would be the discrete value itself.
                    For Box, it fills the array.
                    For Dict/Tuple, applies recursively.

    Returns:
        A sample from the space, filled appropriately.
    """
    if isinstance(space, gym.spaces.Box):
        # Ensure fill_value is compatible with space's dtype and bounds
        val = np.full(space.shape, fill_value, dtype=space.dtype)
        if not space.contains(val):
            # Fallback if fill_value is problematic (e.g. out of bounds)
            # This could happen if fill_value=0 but low bound is >0.
            # A safer bet is to use space.low if bounded.
            if space.is_bounded(): # Check both lower and upper
                 # Create a sample at the lower bound if possible
                 val_low = np.full(space.shape, space.low, dtype=space.dtype)
                 if space.contains(val_low): return val_low
                 # Else, a zero sample if it's contained (common for image-like spaces)
                 zero_sample = np.zeros(space.shape, dtype=space.dtype)
                 if space.contains(zero_sample): return zero_sample
            # If all else fails, sample and hope for the best (or raise error)
            print(f"Warning: fill_value {fill_value} may not be suitable for Box space {space}. Returning a random sample instead.")
            return space.sample()
        return val
    elif isinstance(space, gym.spaces.Discrete):
        if space.contains(fill_value):
            return fill_value
        else: # Fallback to start or 0 if valid
            if space.start is not None and space.contains(space.start): return space.start
            if space.contains(0): return 0
            print(f"Warning: fill_value {fill_value} not in Discrete space {space}. Returning a random sample.")
            return space.sample() # Fallback
    elif isinstance(space, gym.spaces.Dict):
        return {
            key: create_empty_sample(sub_space, fill_value)
            for key, sub_space in space.spaces.items()
        }
    elif isinstance(space, gym.spaces.Tuple):
        return tuple(
            create_empty_sample(sub_space, fill_value)
            for sub_space in space.spaces
        )
    elif isinstance(space, gym.spaces.MultiBinary):
        return np.full(space.shape, fill_value if fill_value in [0,1] else 0, dtype=np.int8)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        # Fill with 0s if 0 is valid for all dimensions, else use start of range.
        # This is complex because each dimension has its own range.
        # Safest is to create an array of starting values for each discrete dimension.
        # space.nvec gives the number of choices for each dimension.
        # space.low and space.high might not be simple scalars.
        # For MultiDiscrete, fill_value is tricky. Let's use 0s if valid.
        sample_val = np.zeros(space.shape, dtype=space.dtype)
        if space.contains(sample_val):
            return sample_val
        # Fallback: create a sample of all lower bounds
        # np.full(space.shape, 0, dtype=space.dtype) is a common default
        # but MultiDiscrete `contains` checks against `nvec`.
        # A sample of all zeros is `[0,0,...,0]` which is usually valid.
        # The `low` and `high` for MultiDiscrete are implicitly 0 and nvec-1 for each component.
        # So, a vector of zeros should be valid.
        return np.zeros(space.nvec.shape, dtype=space.dtype) # Shape of nvec is num_components


    else:
        # For other space types, a random sample might be the only generic way
        print(f"Warning: create_empty_sample not fully implemented for {type(space)}. Returning random sample.")
        return space.sample()

# Example Usage:
# if __name__ == "__main__":
#     # Example Box space
#     box_space = gym.spaces.Box(low=0, high=255, shape=(2, 2), dtype=np.uint8)
#     empty_box = create_empty_sample(box_space, fill_value=0)
#     print("Empty Box sample:\n", empty_box)

#     # Example Discrete space
#     discrete_space = gym.spaces.Discrete(5, start=1) # Values 1,2,3,4,5
#     empty_discrete = create_empty_sample(discrete_space, fill_value=1) # Use start value
#     print("\nEmpty Discrete sample:", empty_discrete)
#     empty_discrete_zero_fail = create_empty_sample(discrete_space, fill_value=0) # 0 not in space
#     print("Empty Discrete sample (fill 0, fallback to start):", empty_discrete_zero_fail)


#     # Example Dict space
#     dict_space = gym.spaces.Dict({
#         "position": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
#         "color_idx": gym.spaces.Discrete(10) # 0-9
#     })
#     empty_dict_sample = create_empty_sample(dict_space, fill_value=0)
#     print("\nEmpty Dict sample:")
#     for key, val in empty_dict_sample.items():
#         print(f"  {key}: {val}")

#     # Example Tuple space
#     tuple_space = gym.spaces.Tuple((
#         gym.spaces.Discrete(3),
#         gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
#     ))
#     empty_tuple_sample = create_empty_sample(tuple_space, fill_value=0)
#     print("\nEmpty Tuple sample:", empty_tuple_sample)

#     # Space size examples
#     print(f"\nSize of Discrete(5): {get_space_size(gym.spaces.Discrete(5))}") # 5
#     print(f"Size of Box(2,2): {get_space_size(box_space)}") # 4
#     print(f"Size of DictSpace: {get_space_size(dict_space)}") # (2*2*sizeof(float)) * 10 ? No, product of discrete counts: inf * 10 = inf
#                                                              # Correct for discrete interpretation: inf (Box) * 10 (Discrete) = inf
#                                                              # If Box interpreted as #elements: 2 (Box elements) * 10 = 20
#                                                              # Current impl: Box is #elements if bounded. Box(-1,1) is bounded. (1-(-1))^2 * 10 = 40?
#                                                              # No, get_space_size for Box is prod(shape) = 2. So 2*10=20.
#     print(f"Size of TupleSpace: {get_space_size(tuple_space)}") # 3 * 1 = 3

#     unbounded_box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
#     print(f"Size of unbounded_box: {get_space_size(unbounded_box)}") # inf
