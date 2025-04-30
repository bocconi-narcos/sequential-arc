# ARC Action Space (`arc_env/action_space.py`)

This document details the `ARCActionSpace` class, which provides a structured action space for reinforcement learning agents interacting with the Abstraction and Reasoning Corpus (ARC) environment (`ARCEnv`).

## Overview

The core challenge in applying RL to ARC is defining an action space that is expressive enough to capture the necessary transformations but structured enough for an agent to learn effectively. `ARCActionSpace` addresses this by:

1. **Composability:** Actions are defined as a pipeline of three sequential sub-actions:
   - **Colour Selection:** Choose a relevant colour from the grid.
   - **Grid Selection:** Select a set of cells (often based on the chosen colour).
   - **Transformation:** Apply an operation to the selected cells.
2. **DSL Integration:** The actual functions for colour selection, grid selection, and transformation are defined externally in a Domain-Specific Language (DSL) located in `arc_env/dsl/`. This promotes modularity and extensibility.
3. **Configuration Presets:** The specific set of available DSL functions for each part of the pipeline is defined by named "presets" in `arc_env/action_config.py`. This allows easy experimentation with different action sets.
4. **Gymnasium Compatibility:** It adheres to the `gymnasium.Space` interface.
5. **Dual Modes:** It supports two primary modes (`factorized` and `joint`) to accommodate different agent architectures.

## Core Concepts

### Action Pipeline

Each action selected by the agent represents a sequence of three function calls executed in order:

1. `colour_fn = ColourSelector.<method>(**kwargs)`
2. `selection_mask = GridSelector.<method>(grid, colour=colour_fn_result, **kwargs)`
3. `new_grid = GridTransformer.<method>(grid, mask=selection_mask, **kwargs)`

The agent chooses a single integer (in `joint` mode) or a dictionary of integers (in `factorized` mode) that corresponds to a pre-defined combination of these functions.

### DSL and Presets

- Primitive operations are implemented in the `ColorSelector`, `GridSelector`, and `GridTransformer` classes in `arc_env/dsl/`.
- `action_config.py` defines a `PRESETS` dictionary specifying available functions per preset.

#### Example Structure in `action_config.py`:

```python
PRESETS = {
    "default": {
        "colour": [
            {"name": "get_colour", "kwargs": {"index": 0}},
            {"name": "get_most_frequent_colour"},
        ],
        "selection": [
            {"name": "select_all"},
            {"name": "select_by_colour"},
        ],
        "transform": [
            {"name": "flip_horizontal"},
            {"name": "change_colour", "kwargs": {"target_colour": 1}},
        ]
    },
    "minimal": {
        # smaller set of actions
    }
}
```

### Modes: factorized vs. joint

- **factorized (default):**
  - Space: `gym.spaces.Dict`
  - Action: `{'colour': 2, 'selection': 5, 'transform': 1}`
  - Use: multi-head RL agents

- **joint:**
  - Space: `gym.spaces.Discrete`
  - Action: `127` (mapped to C, S, T triplet)
  - Use: single-head agents (DQN, tabular)

### Initialization

```python
from arc_env.action_space import ARCActionSpace

# Default preset and factorized mode
action_space_fact = ARCActionSpace()

# Specified preset and joint mode
action_space_joint = ARCActionSpace(preset="minimal", mode="joint")
```

### Parameters

- `preset` (str, default="default")
- `mode` (str, default="factorized")

### Key Methods

- `decode(action)`  
  Returns `(colour_fn, selection_fn, transform_fn)` callable objects

- `encode(colour, selection, transform, key=None)`  
  Returns action representation from function names

- `sample(mask=None)`  
  Returns random valid action

- `contains(x)`  
  Checks if action `x` is valid

- `action_to_str(action)`  
  Returns human-readable string of action pipeline

### Properties

- `sizes`: Dict with counts per category: `{ 'colour': C, 'selection': S, 'transform': T }`
- `mode`: Either "factorized" or "joint"
- `space`: Underlying Gymnasium space

## Example Usage

```python
from arc_env.action_space import ARCActionSpace

action_space = ARCActionSpace(mode="factorized")
random_action = action_space.sample()
print(f"Random Action ({action_space.mode}): {random_action}")

c_fn, s_fn, t_fn = action_space.decode(random_action)
print(f"Decoded functions: {c_fn}, {s_fn}, {t_fn}")

action_str = action_space.action_to_str(random_action)
print(f"Action Description: {action_str}")

try:
    encoded_action = action_space.encode(
        colour="get_most_frequent_colour",
        selection="select_by_colour",
        transform="rotate_cw"
    )
    print(f"Encoded Action ({action_space.mode}): {encoded_action}")

    decoded_again = action_space.decode(encoded_action)
    reencoded_str = action_space.action_to_str(encoded_action)
    print(f"Re-encoded String: {reencoded_str}")

except ValueError as e:
    print(f"Encoding failed: {e}")

print(f"Action Space Sizes: {action_space.sizes}")
```
