# ARC Gymnasium Environment (`arc_env/arc_env.py`)

This document details the `ARCEnv` class, a custom Gymnasium environment designed for applying reinforcement learning to the Abstraction and Reasoning Corpus (ARC).

## Overview

`ARCEnv` provides a standard RL interface (`reset`, `step`) for interacting with ARC tasks. It handles:

1.  **Task Loading:** Loading ARC task definitions from JSON files (compatible with the official ARC dataset format).
2.  **State Representation:** Representing the environment state using NumPy arrays, typically including the current grid and the target grid, padded to a fixed canvas size.
3.  **Action Handling:** Accepting actions defined by an `ARCActionSpace` instance and applying the corresponding transformations (colour selection → grid selection → transformation) to the current grid.
4.  **Reward Calculation:** Computing rewards based on the similarity (maximum overlap) between the transformed grid and the target grid, incorporating penalties for inefficient steps or undesirable outcomes.
5.  **Episode Management:** Handling episode termination (task solved) and truncation (e.g., grid becomes uniform, potentially indicating a dead end).
6.  **Rendering:** Providing visualizations of the state transitions for debugging and analysis.
7.  **Sequence Evaluation:** Offering a utility method to test a fixed sequence of actions on all pairs (train/test) of a specific ARC task.

## Core Concepts

### Gymnasium Interface

`ARCEnv` inherits from `gymnasium.Env` and implements the standard API:

* `reset()`: Starts a new episode, samples an ARC task (or a specific one if requested), and returns the initial observation.
* `step(action)`: Executes one action in the environment, updates the state, calculates the reward, and determines if the episode has ended. Returns `(observation, reward, terminated, truncated, info)`.
* `render()`: Displays a visualization of the last step (requires `mode='human'`).
* `close()`: (Inherited) Performs cleanup if needed.

### State and Observation Space

* **Internal State (`self.state`):** A NumPy array representing the current situation. By default, it has the shape `(canvas_height, canvas_width, 2)` and `dtype=np.int8`.
    * `self.state[..., 0]`: The *current* working grid, padded to `(canvas_height, canvas_width)`. This is the grid modified by the agent's actions.
    * `self.state[..., 1]`: The *target* grid for the current task pair, also padded to the canvas size. This generally remains fixed throughout an episode.
* **Observation Space (`self.observation_space`):** A `gymnasium.spaces.Box` instance describing the structure of the observations returned by `reset` and `step`. It matches the shape and data type of `self.state`. The values range from 0 to 9 (standard ARC colours).
* **Padding:** Input and target grids from the ARC tasks are smaller than the canvas. They are padded (using `dsl.utils.padding.pad_grid`) to the fixed `canvas_size` before being placed in the state. Transformations operate on the padded grid. For reward calculation and checking termination, the working grid is unpadded (`dsl.utils.padding.unpad_grid`).

### Action Space

The environment requires an instance of `ARCActionSpace` during initialization. This object defines the set of possible actions the agent can take. See [docs/action_space.md](docs/action_space.md) for details on how actions are structured and decoded.

### Reward Calculation (`_maximum_overlap`)

The reward function incentivizes the agent to make the current grid (`self.state[..., 0]`) identical to the target grid (`self.state[..., 1]`).

* **Primary Reward:** Based on the change in "maximum overlap" between the current grid and the target grid.
    * `_maximum_overlap(arr1, arr2)` calculates the best possible alignment score by sliding `arr2` over `arr1` without rotation. It returns the fraction of matching cells in the best alignment relative to the size of `arr2`.
    * The reward for a step is proportional to: `_maximum_overlap(new_grid, target) - _maximum_overlap(previous_grid, target)`. This difference is scaled (e.g., by 100).
* **Penalties:**
    * `step_penalty`: A small negative reward applied on steps that do *not* increase the overlap, encouraging progress.
    * `shape_penalty`: An additional penalty if the step doesn't increase overlap *and* the resulting grid shape doesn't match the target shape.
    * `no_change_penalty`: A larger penalty if the action results in *no change* to the grid, discouraging redundant actions.
    * `trunc_penalty`: A large penalty applied if the episode is truncated (e.g., grid becomes uniform) but not solved.
* **Bonus:**
    * `completion_bonus`: A positive reward added when the task is successfully solved (current grid exactly matches the target grid).

### Termination and Truncation

* **Termination (`terminated = True`):** The episode ends successfully if the unpadded current grid becomes identical in shape and content to the unpadded target grid.
* **Truncation (`truncated = True`):** The episode ends prematurely if a condition indicating a likely dead end is met. The current implementation truncates if the grid becomes monochromatic (all cells have the same colour), unless this state is also the target state.

### Rendering

The `render(mode='human')` method uses `matplotlib` to display a visualization of the *last completed step*. It shows:

1.  **Before:** The grid state before the action was applied.
2.  **Mask:** The selection mask generated by the action's selection function, highlighting the area targeted by the transform. An inset shows the colour chosen by the colour selection function.
3.  **After:** The grid state after the action's transformation was applied.
4.  **Target:** The target grid for the episode.

The figure title includes the human-readable action string, the reward received for that step, and the ARC challenge key.

## Initialization

```python
from arc_env.action_space import ARCActionSpace
from arc_env.arc_env import ARCEnv

# 1. Create an action space instance
action_space = ARCActionSpace(preset="default", mode="factorized")

# 2. Initialize the environment
env = ARCEnv(
    challenges_json="path/to/arc/data/training_challenges.json",
    solutions_json="path/to/arc/data/training_solutions.json", # Needed for solvers, can be same as challenges
    action_space=action_space,
    canvas_size=30,         # Max dimensions for padding
    step_penalty=1,
    shape_penalty=1,
    no_change_penalty=5,
    trunc_penalty=100,
    completion_bonus=25,
    seed=42                 # For reproducibility
)
```

Parameters:

- **challenges_json** (`str | Path`): Path to the JSON file containing ARC challenge definitions (e.g., `training_challenges.json`).
- **solutions_json** (`str | Path`): Path to the JSON file containing ARC solution definitions (e.g., `training_solutions.json`). Currently used by `evaluate_sequence` but could be integrated further.
- **action_space** (`ARCActionSpace`): An initialized instance of the action space the agent will use.
- **canvas_size** (`int`, default=`30`): The height and width to which grids will be padded. Should be large enough for the biggest grids in the dataset.
- **step_penalty** (`int`, default=`1`): Penalty for non-improving steps.
- **shape_penalty** (`int`, default=`1`): Additional penalty if shape is wrong on non-improving steps.
- **no_change_penalty** (`int`, default=`5`): Penalty for actions that don't change the grid.
- **trunc_penalty** (`int`, default=`100`): Penalty upon truncation.
- **completion_bonus** (`int`, default=`25`): Reward upon successful termination.
- **max_branch** (`int`, default=`1`): (Currently unused in step) Parameter potentially for future use with frontier/search mechanisms.
- **seed** (`int | None`, default=`None`): Seed for the environment's internal random number generator (`numpy.random.default_rng`), ensuring reproducibility of task sampling if set.

Key Methods

**reset(*, seed=None, options=None):**

Resets the environment for a new episode.

- **seed**: Optionally re-seed the environment's RNG.
- **options** (`dict | None`): Allows specifying which task to load:
  - `options['key']` (`str`): Force loading a specific ARC challenge key.
  - `options['min_examples']` (`int`): Sample only from challenges with at least this many training pairs.
  - `options['index']` (`int`): Force loading a specific training pair index within the chosen challenge key.

Returns: `(initial_observation, info_dict)`. The `info_dict` is currently empty upon reset but used in step.

**step(action):**

Applies the given action (in the format expected by the `action_space`, i.e., `dict` or `int`).

- Decodes the action using `action_space.decode`.
- Executes the colour → selection → transform pipeline.
- Calculates reward, terminated, truncated status.
- Updates `self.state` and `self.info`.

Returns: `(observation, reward, terminated, truncated, info)` where `info` contains diagnostic data like `{ "key": str, "actions": list, "action_desc": list, "num_actions": int, "solved": bool }`.

**render(*, mode='human'):**

Displays the visualization of the last step taken. Only `mode='human'` is supported. Must be called after `step`.

**evaluate_sequence(challenge_key, actions, *, render=True, exclude=(), include=None, stepwise=False):**

A utility method to test a fixed actions sequence on a specific `challenge_key`.

- It iterates through all training pairs (and test pairs if available and implemented) of the challenge.
- For each pair, it resets the environment to that pair's input/target and executes the full actions sequence.

Arguments:
- `challenge_key`: The ARC task key (e.g., `"1e0a9b12"`).
- `actions`: A list or array of actions encoded according to the environment's `action_space`.
- `render`: If `True`, generates and saves a summary plot (`solvers/renders/solver_{challenge_key}.png`).
- `exclude`: An iterable of pair indices to skip.
- `include`: An iterable of pair indices to only run.
- `stepwise`: If `True`, calls `render()` after each step (requires `render=True`).

Returns: A list of dictionaries, one per evaluated pair, containing `{ 'before', 'after', 'target', 'reward', 'solved', 'pair_idx' }`.

Example Usage (Basic RL Loop)

```python
import gymnasium as gym
from arc_env.action_space import ARCActionSpace
from arc_env.arc_env import ARCEnv
import numpy as np

# --- Setup ---
action_space = ARCActionSpace(mode="factorized") # Or joint
env = ARCEnv(
    challenges_json="path/to/training_challenges.json",
    solutions_json="path/to/training_solutions.json",
    action_space=action_space,
    seed=0
)

# --- Run one episode ---
observation, info = env.reset(options={"key": "1a74a9c3", "index": 0}) # Load specific task pair
terminated = False
truncated = False
total_episode_reward = 0
step_count = 0
max_steps = 50 # Set a limit

while not terminated and not truncated and step_count < max_steps:
    # Replace with your agent's policy
    # For now, use random actions
    action = env.action_space.sample()

    # Take action
    observation, reward, terminated, truncated, info = env.step(action)

    # Render the step (optional)
    env.render(mode='human')
    # input("Press Enter to continue...") # Pause for viewing

    total_episode_reward += reward
    step_count += 1
    print(f"Step: {step_count}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")


print(f"\nEpisode Finished.")
print(f"Total Reward: {total_episode_reward:.2f}")
print(f"Solved: {info.get('solved', False)}")
print(f"Final Info: {info}")

env.close()
```
