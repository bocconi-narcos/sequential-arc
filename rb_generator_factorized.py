"""
Generates a replay buffer of (state, action, reward, next_state, done) transitions
from the Abstract Reasoning Corpus (ARC) using a factorized action space.

This script interacts with an ARC environment (`env.ARCEnv`), loads ARC tasks
from specified JSON files, filters tasks based on grid dimensions, and then
collects interaction data by taking random actions in the environment.

The primary inputs are:
- Path to a challenges JSON file (e.g., `dataset/evaluation/challenges.json`).
- Path to a solutions JSON file (e.g., `dataset/evaluation/solutions.json`).

The script outputs a single pickle file containing the replay buffer,
which is a list of transition tuples. Each tuple contains:
(unpadded_state_list, action_dict, reward_float, unpadded_next_state_list, done_bool)
where action_dict is a dictionary with keys "colour", "selection", and "transform".
"""
import argparse
import pickle
import numpy as np
import os
import random
import json
from collections import deque
from typing import List, Dict, Any

from env import ARCEnv
from action_space import ARCActionSpace
from dsl.utils.padding import unpad_grid, pad_grid
from dsl.utils.background import find_background_colour
from dsl.utils.random_grid import generate_random_grid
from pathlib import Path

# adjust paths
DATA_DIR = Path(__file__).resolve().parent / "data"
CHALLENGES_JSON = DATA_DIR / "challenges.json"
SOLUTIONS_JSON  = DATA_DIR / "solutions.json"

# For the replay buffer, locally
OUTPUT_FILEPATH = Path(__file__).resolve().parent / "replay_buffer_factorized.pkl"


def load_and_filter_grids(solutions_data, challenges_data, max_h, max_w):
    """
    Filters grids from solutions_data and challenges_data based on maximum height and width.
    Returns a list of (task_key, example_index) for valid grids.
    """
    valid_grids = []
    
    for task_key in challenges_data:
        if 'train' in challenges_data[task_key]:
            for i, example in enumerate(challenges_data[task_key]['train']):
                try:
                    # Check both input and output dimensions
                    input_grid = np.array(example['input'])
                    output_grid = np.array(solutions_data[task_key])  # Solution is directly in solutions_data
                    
                    if (input_grid.shape[0] <= max_h and input_grid.shape[1] <= max_w and
                        output_grid.shape[0] <= max_h and output_grid.shape[1] <= max_w):
                        valid_grids.append((task_key, i))
                except Exception as e:
                    print(f"Warning: Could not parse grid for task {task_key}, example {i}: {e}")
                    continue
    
    return valid_grids

def process_state(state: np.ndarray) -> np.ndarray:
    """
    Process a state array to ensure it's 2D. If it's 3D (from selection functions),
    combine all layers into a single 2D array.
    
    Args:
        state: Input state array (2D or 3D)
        
    Returns:
        2D numpy array
    """
    if state.ndim == 3:
        # If we have multiple layers (from selection functions),
        # combine them by taking the maximum value across layers
        return np.max(state, axis=0)
    return state

def count_unique_colors(grid: np.ndarray) -> int:
    """Return the number of unique colors in the grid."""
    return len(np.unique(grid))


def get_selection_mask(action_space: ARCActionSpace, action: Dict[str, int], grid: np.ndarray) -> np.ndarray:
    """
    Given an action and a grid, return the boolean selection mask for the action.
    """
    _, selection_fn, _ = action_space.decode(action)
    # Some selectors require a color argument, others do not
    try:
        mask = selection_fn(grid, action["colour"])
    except TypeError:
        mask = selection_fn(grid)
    # Ensure mask is 2D boolean
    if mask.ndim == 3:
        mask = np.any(mask, axis=0)
    return mask.astype(bool)


def validate_colors(grid: np.ndarray, name: str = "grid") -> None:
    """
    Validate that all colors in a grid are within the valid ARC range [0, 9].
    """
    if grid.min() < 0 or grid.max() > 9:
        raise ValueError(f"Colors in {name} are outside valid ARC range [0, 9]. "
                        f"Found range [{grid.min()}, {grid.max()}]")


def create_transition_dict(
    state: np.ndarray,
    target_state: np.ndarray,
    action: Dict[str, int],
    action_space: ARCActionSpace,
    reward: float,
    next_state: np.ndarray,
    done: bool,
    info: Dict[str, Any],
    canvas_size: int = 10,
) -> Dict[str, Any]:
    """
    Create a transition dictionary matching the required schema, padding all arrays to (canvas_size, canvas_size).
    """
    # Validate colors before padding
    validate_colors(state, "state")
    validate_colors(target_state, "target_state")
    validate_colors(next_state, "next_state")
    
    # Pad all relevant arrays
    state_padded = pad_grid(state, (canvas_size, canvas_size))
    target_state_padded = pad_grid(target_state, (canvas_size, canvas_size))
    next_state_padded = pad_grid(next_state, (canvas_size, canvas_size))
    selection_mask = get_selection_mask(action_space, action, state)
    selection_mask_padded = pad_grid(selection_mask.astype(np.int8), (canvas_size, canvas_size)).astype(bool)

    # Determine the ARC colour: most common in selection, or background if empty
    selected_colors = state[selection_mask]
    if selected_colors.size == 0:
        arc_colour = find_background_colour(state)
    else:
        arc_colour = np.bincount(selected_colors).argmax()

    return {
        "state": state_padded,
        "target_state": target_state_padded,
        "color_in_state": count_unique_colors(state),
        "action": {
            "colour": action["colour"],
            "selection": action["selection"],
            "transform": action["transform"],
        },
        "colour": int(arc_colour),
        "selection_mask": selection_mask_padded,
        "reward": float(reward),
        "next_state": next_state_padded,
        "done": bool(done),
        "info": info.copy() if info is not None else {},
    }


def generate_buffer_from_challenges(
    env: ARCEnv,
    action_space: ARCActionSpace,
    valid_initial_grids: List,
    buffer_size: int,
    num_steps_per_grid: int,
    canvas_size: int,
) -> List[Dict[str, Any]]:
    """
    Generate a replay buffer from the challenge dataset, padding all grids to (canvas_size, canvas_size).
    Always call env.reset() at the start of each episode.
    """
    buffer = []
    transitions_added = 0
    while len(buffer) < buffer_size:
        task_key, example_index = random.choice(valid_initial_grids)
        print(f"\n[Challenge] Starting new episode: task_key={task_key}, example_index={example_index}, buffer size={len(buffer)}/{buffer_size}")
        example = env._challenges[task_key]['train'][example_index]
        input_grid = np.array(example['input'])
        output_grid = np.array(env._solutions[task_key])
        # Always reset the environment at the start of each episode
        try:
            obs, info = env.reset(options={"input_grid": input_grid, "target_grid": output_grid})
        except TypeError:
            # If env.reset does not accept options, just call reset()
            obs, info = env.reset()
        # Extract the initial state and target state from the observation
        state = unpad_grid(obs[..., 0])
        target_state = unpad_grid(obs[..., 1])
        for step in range(num_steps_per_grid):
            if len(buffer) >= buffer_size:
                break
            action = action_space.sample()
            try:
                next_observation, reward, terminated, truncated, _info = env.step(action)
                done = terminated or truncated
                next_state = unpad_grid(next_observation[..., 0])
                transition = create_transition_dict(
                    state=state,
                    target_state=target_state,
                    action=action,
                    action_space=action_space,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    info={},
                    canvas_size=canvas_size,
                )
                buffer.append(transition)
                transitions_added += 1
                if transitions_added % 10 == 0 or transitions_added == buffer_size:
                    print(f"[Challenge] Added {transitions_added}/{buffer_size} transitions to buffer.")
                state = next_state.copy()
                if done:
                    break
            except Exception as e:
                continue
    print(f"[Challenge] Buffer generation complete. Total transitions: {len(buffer)}")
    return buffer


def generate_buffer_from_random(
    action_space: ARCActionSpace,
    buffer_size: int,
    grid_shape: tuple,
    num_colors: int,
    rng: np.random.Generator = None,
    lambda_poisson: float = 7.0,
    canvas_size: int = 10,
) -> List[Dict[str, Any]]:
    """
    Generate a replay buffer from random grids and random actions, padding all grids to (canvas_size, canvas_size).
    """
    if rng is None:
        rng = np.random.default_rng()
    buffer = []
    transitions_added = 0
    while len(buffer) < buffer_size:
        # 1. Generate initial random grid
        state = generate_random_grid(grid_shape, num_colors, rng)
        n = rng.poisson(lambda_poisson)
        n = max(1, n)
        print(f"\n[Random] Starting new episode: grid_shape={grid_shape}, num_colors={num_colors}, episode_length={n}, buffer size={len(buffer)}/{buffer_size}")
        # 3. Apply n random actions to get target_state
        temp_grid = state.copy()
        for _ in range(n):
            action = action_space.sample()
            _, selection_fn, transform_fn = action_space.decode(action)
            try:
                selection_mask = get_selection_mask(action_space, action, temp_grid)
                temp_grid = transform_fn(temp_grid, selection_mask)
            except Exception:
                continue
        target_state = temp_grid.copy()
        # 4. Roll out the episode, storing transitions
        current_grid = state.copy()
        for step in range(n):
            if len(buffer) >= buffer_size:
                break
            action = action_space.sample()
            _, selection_fn, transform_fn = action_space.decode(action)
            try:
                selection_mask = get_selection_mask(action_space, action, current_grid)
                next_grid = transform_fn(current_grid, selection_mask)
                reward = 0.0  # You may define a reward function if needed
                done = (step == n - 1)
                info = {"step_distance_to_target": n - step - 1}
                transition = create_transition_dict(
                    state=current_grid,
                    target_state=target_state,
                    action=action,
                    action_space=action_space,
                    reward=reward,
                    next_state=next_grid,
                    done=done,
                    info=info,
                    canvas_size=canvas_size,
                )
                buffer.append(transition)
                transitions_added += 1
                if transitions_added % 10 == 0 or transitions_added == buffer_size:
                    print(f"[Random] Added {transitions_added}/{buffer_size} transitions to buffer.")
                current_grid = next_grid.copy()
            except Exception:
                continue
    print(f"[Random] Buffer generation complete. Total transitions: {len(buffer)}")
    return buffer


def main():
    """
    Main function to generate the ARC replay buffer with factorized action space.
    Supports both challenge dataset and random grid generation.
    """
    parser = argparse.ArgumentParser(description="Generate a replay buffer for ARC with factorized action space.")
    parser.add_argument("--mode", type=str, choices=["challenge", "random"], default="challenge",
                        help="Data source: 'challenge' for dataset, 'random' for random grids.")
    parser.add_argument("--output_filepath", type=str, default=str(Path(__file__).resolve().parent / "replay_buffer_factorized.pkl"),
                        help="Where to save the replay buffer.")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Max transitions in the buffer.")
    parser.add_argument("--num_steps_per_grid", type=int, default=5, help="Number of random actions per grid (challenge mode).")
    parser.add_argument("--canvas_size", type=int, default=10, help="Canvas size for ARCEnv.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--action_preset", type=str, default="default", help="Action preset for ARCActionSpace.")
    # Challenge mode args
    parser.add_argument("--challenges_json_path", type=str, default=str(Path(__file__).resolve().parent / "data/challenges.json"),
                        help="Path to ARC challenges JSON.")
    parser.add_argument("--solutions_json_path", type=str, default=str(Path(__file__).resolve().parent / "data/solutions.json"),
                        help="Path to ARC solutions JSON.")
    parser.add_argument("--max_grid_dim_h", type=int, default=5, help="Max height for initial grids (challenge mode).")
    parser.add_argument("--max_grid_dim_w", type=int, default=5, help="Max width for initial grids (challenge mode).")
    # Random mode args
    parser.add_argument("--random_grid_h", type=int, default=5, help="Height of random grid (random mode).")
    parser.add_argument("--random_grid_w", type=int, default=5, help="Width of random grid (random mode).")
    parser.add_argument("--random_num_colors", type=int, default=4, help="Number of colors for random grid (random mode).")
    parser.add_argument("--random_lambda_poisson", type=float, default=7.0, help="Lambda for Poisson episode length (random mode).")
    args = parser.parse_args()

    # Validate random_num_colors
    if args.random_num_colors > 10:
        raise ValueError("--random_num_colors must be <= 10 (ARC palette constraint)")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
    else:
        rng = np.random.default_rng()

    action_space = ARCActionSpace(preset=args.action_preset, mode="factorized")

    if args.mode == "challenge":
        # Load solutions and challenges data
        with open(args.solutions_json_path, 'r') as f:
            solutions_data = json.load(f)
        with open(args.challenges_json_path, 'r') as f:
            challenges_data = json.load(f)
        env = ARCEnv(
            challenges_json=args.challenges_json_path,
            solutions_json=args.solutions_json_path,
            action_space=action_space,
            canvas_size=args.canvas_size,
            seed=args.seed
        )
        valid_initial_grids = []
        for task_key in challenges_data:
            if 'train' in challenges_data[task_key]:
                for i, example in enumerate(challenges_data[task_key]['train']):
                    input_grid = np.array(example['input'])
                    output_grid = np.array(solutions_data[task_key])
                    if (input_grid.shape[0] <= args.max_grid_dim_h and input_grid.shape[1] <= args.max_grid_dim_w and
                        output_grid.shape[0] <= args.max_grid_dim_h and output_grid.shape[1] <= args.max_grid_dim_w):
                        valid_initial_grids.append((task_key, i))
        if not valid_initial_grids:
            print(f"No initial grids found matching criteria (max_h={args.max_grid_dim_h}, max_w={args.max_grid_dim_w}). Exiting.")
            return
        buffer = generate_buffer_from_challenges(
            env=env,
            action_space=action_space,
            valid_initial_grids=valid_initial_grids,
            buffer_size=args.buffer_size,
            num_steps_per_grid=args.num_steps_per_grid,
            canvas_size=args.canvas_size,
        )
    else:
        buffer = generate_buffer_from_random(
            action_space=action_space,
            buffer_size=args.buffer_size,
            grid_shape=(args.random_grid_h, args.random_grid_w),
            num_colors=args.random_num_colors,
            rng=rng,
            lambda_poisson=args.random_lambda_poisson,
            canvas_size=args.canvas_size,
        )
    # Save the buffer
    with open(args.output_filepath, 'wb') as f:
        pickle.dump(buffer, f)
    print(f"Replay buffer saved to {args.output_filepath}. Total transitions: {len(buffer)}")

if __name__ == "__main__":
    main() 