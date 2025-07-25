# All settings are loaded from that file. See the example in the repo for details.
#
# Usage: python rb_generator_factorized.py

import os
import pickle
import numpy as np
import random
import json
from collections import deque
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import yaml
import torch

# adjust paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHALLENGES_JSON = DATA_DIR / "challenges.json"
SOLUTIONS_JSON  = DATA_DIR / "solutions.json"

# For the replay buffer, locally
OUTPUT_FILEPATH = Path(__file__).resolve().parent / "replay_buffer_factorized.pkl"

# Add project root to Python path for imports
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from env import ARCEnv
from action_space import ARCActionSpace
from dsl.utils.padding import unpad_grid, pad_grid
from dsl.utils.background import find_background_colour
from dsl.utils.random_grid import generate_random_grid

from buffer.utils import (load_and_filter_grids, process_state, count_unique_colors,
                   get_selection_mask, validate_colors, count_unique_colors_exclude_padding, 
                     most_present_color_exclude_padding, least_present_color_exclude_padding, get_grid_shape,
                     validate_grid_padding)




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
    Additional fields:
        - shape: (height, width) of the unpadded state grid
        - num_colors_grid: number of unique colors in the unpadded state grid (excluding padding)
        - most_present_color: most frequent color in the unpadded state grid (excluding padding, -1 if empty)
    """
    # Validate colors before padding
    validate_colors(state, "state")
    validate_colors(target_state, "target_state")
    validate_colors(next_state, "next_state")
    
    # Validate input and output grids
    if state.shape[0] > canvas_size or state.shape[1] > canvas_size:
        raise ValueError(
            f"State exceeds canvas size {canvas_size}: "
            f"shape={state.shape}"
        )
    
    if next_state.shape[0] > canvas_size or next_state.shape[1] > canvas_size:
        raise ValueError(
            f"Next state exceeds canvas size {canvas_size}: "
            f"shape={next_state.shape}"
        )

    # Pad all relevant arrays
    state_padded                    = pad_grid(state, (canvas_size, canvas_size))
    target_state_padded             = pad_grid(target_state, (canvas_size, canvas_size))
    next_state_padded               = pad_grid(next_state, (canvas_size, canvas_size))
    selection_mask                  = get_selection_mask(action_space, action, state)
    selection_mask_padded           = pad_grid(selection_mask.astype(np.int8), (canvas_size, canvas_size))

    # Determine the ARC colour: output of the color selection function
    colour_fn, _, _                 = action_space.decode(action)
    arc_colour                      = colour_fn(state)

    # Compute new fields from the unpadded state
    shape                           = get_grid_shape(state)
    num_colors_grid                 = count_unique_colors_exclude_padding(state)
    most_present                    = most_present_color_exclude_padding(state)
    least_present                   = least_present_color_exclude_padding(state)

    shape_next                     = get_grid_shape(next_state)
    num_colors_grid_next           = count_unique_colors_exclude_padding(next_state)
    most_present_next              = most_present_color_exclude_padding(next_state)
    least_present_next             = least_present_color_exclude_padding(next_state)

    shape_target                   = get_grid_shape(target_state)
    num_colors_grid_target         = count_unique_colors_exclude_padding(target_state)
    most_present_target            = most_present_color_exclude_padding(target_state)
    least_present_target           = least_present_color_exclude_padding(target_state)

    transition = {
        "state": state_padded,
        "target_state": target_state_padded,
        "action": {
            "colour": action["colour"],
            "selection": action["selection"],
            "transform": action["transform"],
        },
        
        "colour": int(arc_colour),
        "selection_mask": selection_mask_padded,
        "reward": float(reward),
        "shape": shape,
        "shape_target": shape_target,
        "shape_next": shape_next,
        
        "num_colors_grid": num_colors_grid,
        "most_present_color": most_present,
        "least_present_color": least_present,
        "num_colors_grid_target": num_colors_grid_target,
        "most_present_color_target": most_present_target,
        "least_present_color_target": least_present_target,
        "num_colors_grid_next": num_colors_grid_next,
        "most_present_color_next": most_present_next,
        "least_present_color_next": least_present_next,

        "next_state": next_state_padded,
        "done": bool(done),
        "info": info.copy() if info is not None else {},
    }

    # Validate padding for state, target_state, and next_state
    validate_grid_padding(transition["state"], state.shape, canvas_size, "state")
    validate_grid_padding(transition["target_state"], target_state.shape, canvas_size, "target_state")
    validate_grid_padding(transition["next_state"], next_state.shape, canvas_size, "next_state")

    return transition


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
    buffer                          = []
    transitions_added               = 0

    while len(buffer) < buffer_size:
        task_key, example_index     =   random.choice(valid_initial_grids)
        
        print(f"\n[Challenge] Starting new episode: task_key={task_key}, example_index={example_index}, buffer size={len(buffer)}/{buffer_size}")

        example                     = env._challenges[task_key]['train'][example_index]
        input_grid                  = np.array(example['input'])
        output_grid                 = np.array(env._solutions[task_key])

        # Always reset the environment at the start of each episode
        try:
            obs, info               = env.reset(options={"input_grid": input_grid, "target_grid": output_grid})
        
        except TypeError:
            # If env.reset does not accept options, just call reset()
            obs, info               = env.reset()
        
        # Extract the initial state and target state from the observation
        state                       = unpad_grid(obs[..., 0])
        target_state                = unpad_grid(obs[..., 1])
        
        for step in range(num_steps_per_grid):
            if len(buffer) >= buffer_size:
                break
            action                  = action_space.sample()

            
            next_observation, reward, terminated, truncated, _info = env.step(action)
            done                = terminated or truncated
            next_state          = unpad_grid(next_observation[..., 0])

            
            info                = {"transition_type": "challenge"}
            
            transition          = create_transition_dict(
                state=state,
                target_state=target_state,
                action=action,
                action_space=action_space,
                reward=reward,
                next_state=next_state,
                done=done,
                info=info,
                canvas_size=canvas_size,
            )

            buffer.append(transition)
            transitions_added  += 1
            
            if transitions_added % 10 == 0 or transitions_added == buffer_size:
                print(f"[Challenge] Added {transitions_added}/{buffer_size} transitions to buffer.")
            
            state               = next_state.copy()
            
            if done:
                break

    print(f"[Challenge] Buffer generation complete. Total transitions: {len(buffer)}")
    return buffer


def generate_buffer_from_random(
    env: ARCEnv,
    action_space: ARCActionSpace,
    buffer_size: int,
    grid_shape: tuple,
    num_colors: int,
    lambda_poisson: float = 7.0,
    canvas_size: int = 10,
) -> List[Dict[str, Any]]:
    """
    Generate a replay buffer from random grids and random actions, padding all grids to (canvas_size, canvas_size).
    """

    buffer                          = []
    transitions_added               = 0
    while len(buffer) < buffer_size:
        num_colors                  = np.clip(np.random.poisson(3) + 2, 2, 10)
        state                       = generate_random_grid(grid_shape, num_colors)
        n                           = np.random.poisson(lambda_poisson)
        n                           = max(1, n)
        print(f"\n[Random] Starting new episode: grid_shape={grid_shape}, num_colors={num_colors}, episode_length={n}, buffer size={len(buffer)}/{buffer_size}")
        
        # Reset environment to get a clean state
        obs, info                   = env.reset()

        # 2. Generate the sequence of actions that will lead to the target state
        action_sequence             = []
        temp_grid                   = state.copy()
        
        for _ in range(n):
            action                  = action_space.sample()
            action_sequence.append(action)
            try:
                _, selection_fn, transform_fn = action_space.decode(action)
                selection_mask      = get_selection_mask(action_space, action, temp_grid)
                temp_grid           = transform_fn(temp_grid, selection_mask)
            except Exception:
                continue
        
        target_state                = temp_grid.copy()
        
        # 3. Roll out the episode using the exact same action sequence, storing transitions
        current_grid                = state.copy()
        action_idx                  = 0
        for step in range(n):
            if len(buffer) >= buffer_size or action_idx >= len(action_sequence):
                break
            
            # Use the pre-generated action from the sequence
            action                  = action_sequence[action_idx]
            action_idx              += 1
            
            try:
                next_observation, reward, terminated, truncated, _info = env.step(action)
                selection_mask      = get_selection_mask(action_space, action, current_grid)
                next_state          = unpad_grid(next_observation[..., 0])
                done                = (step == n - 1)
                # Correct step_distance_to_target: 1 for last step, 2 for penultimate, etc.
                step_distance       = n - step
                info                = {"step_distance_to_target": step_distance, "transition_type": "random"}
                
                transition          = create_transition_dict(
                    state=current_grid,
                    target_state=target_state,
                    action=action,
                    action_space=action_space,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    info=info,
                    canvas_size=canvas_size,
                )
                
                buffer.append(transition)
                transitions_added   += 1
                
                if transitions_added % 10 == 0 or transitions_added == buffer_size:
                    print(f"[Random] Added {transitions_added}/{buffer_size} transitions to buffer.")
                
                current_grid        = next_state.copy()
                
                if done:
                    break
            except Exception:
                continue

    print(f"[Random] Buffer generation complete. Total transitions: {len(buffer)}")
    return buffer

def generate_buffer_mixed(
    env: ARCEnv,
    action_space: ARCActionSpace,
    valid_initial_grids: List,
    buffer_size: int,
    num_steps_per_grid: int,
    canvas_size: int,
    grid_shape_lambda: tuple,
    num_colors: int,
    lambda_poisson: float = 7.0,
    skip_no_change_steps: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate a replay buffer from a mix of random grids and actual ARC challenges and random actions, padding all grids to (canvas_size, canvas_size).
    
    Args:
        skip_no_change_steps: If True, steps that don't change the state don't count towards num_steps_per_grid
    """

    buffer                          = []
    transitions_added               = 0
    row_lambda                  = grid_shape_lambda[0]
    col_lambda                  = grid_shape_lambda[1]
    
    while len(buffer) < buffer_size:
        # Randomly choose between challenge and random (3/97)
        use_challenge               = random.random() < 0.03
        if use_challenge:
            # --- Challenge episode (like generate_buffer_from_challenges) ---
            task_key, example_index = random.choice(valid_initial_grids)
            
            print(f"\n[Challenge] Starting new episode: task_key={task_key}, example_index={example_index}, buffer size={len(buffer)}/{buffer_size}")
            
            example                 = env._challenges[task_key]['train'][example_index]
            input_grid              = np.array(example['input'])
            output_grid             = np.array(env._solutions[task_key])
            # Always reset the environment at the start of each episode
            try:
                obs, info           = env.reset(options={"input_grid": input_grid, "target_grid": output_grid})
            except TypeError:
                # If env.reset does not accept options, just call reset()
                obs, info           = env.reset()
            
            # Extract the initial state and target state from the observation
            state                   = unpad_grid(obs[..., 0])
            target_state            = unpad_grid(obs[..., 1])
            
            step                    = 0
            while step < num_steps_per_grid:
                if len(buffer) >= buffer_size:
                    break
                action              = action_space.sample()
                
                next_observation, reward, terminated, truncated, _info = env.step(action)
   
                done            = terminated or truncated
                next_state      = unpad_grid(next_observation[..., 0])
                
                # Check if state changed when skip_no_change_steps is enabled
                state_changed = not np.array_equal(state, next_state)
                
                info            = {"transition_type": "challenge"}
                transition      = create_transition_dict(
                    state=state,
                    target_state=target_state,
                    action=action,
                    action_space=action_space,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    info=info,
                    canvas_size=canvas_size,
                )
                buffer.append(transition)
                transitions_added += 1
                
                if transitions_added % 10 == 0 or transitions_added == buffer_size:
                    print(f"[Challenge] Added {transitions_added}/{buffer_size} transitions to buffer.")
                state = next_state.copy()
                
                # Only increment step counter if state changed or if skip_no_change_steps is False
                if state_changed or not skip_no_change_steps:
                    step += 1
                
                if done:
                    break


        else:
            # --- Random episode (like generate_buffer_from_random) ---
            # 1. Generate initial random grid
            num_rows                = np.clip(np.random.poisson(row_lambda) + 2, 2, 10)
            num_cols                = np.clip(np.random.poisson(col_lambda) + 2, 2, 10)
            grid_shape              = (num_rows, num_cols)

            num_colors              = np.clip(np.random.poisson(3) + 2, 2, 10)
            state                   = generate_random_grid(grid_shape, num_colors)
            n                       = np.random.poisson(lambda_poisson)
            n                       = max(1, n)
            print(f"\n[Random] Starting new episode: grid_shape={grid_shape}, num_colors={num_colors}, episode_length={n}, buffer size={len(buffer)}/{buffer_size}")
            # Reset environment to get a clean state
            obs, info               = env.reset()
            
            # 2. Generate the sequence of actions that will lead to the target state
            action_sequence         = []
            temp_grid               = state.copy()
            for _ in range(n):
                action              = action_space.sample()
                action_sequence.append(action)
                try:
                    _, selection_fn, transform_fn = action_space.decode(action)
                    selection_mask  = get_selection_mask(action_space, action, temp_grid)
                    temp_grid       = transform_fn(temp_grid, selection_mask)
                except Exception:
                    # If action fails, just continue with current grid
                    continue

            target_state            = temp_grid.copy()
            
            # 3. Roll out the episode using the exact same action sequence, storing transitions
            current_grid            = state.copy()
            action_idx              = 0
            step                    = 0
            while step < n and action_idx < len(action_sequence):
                if len(buffer) >= buffer_size:
                    break
                
                # Use the pre-generated action from the sequence
                action              = action_sequence[action_idx]
                action_idx          += 1
                
                try:
                    next_observation, reward, terminated, truncated, _info = env.step(action)
                    selection_mask  = get_selection_mask(action_space, action, current_grid)
                    next_state      = unpad_grid(next_observation[..., 0])
                    
                    # Check if state changed when skip_no_change_steps is enabled
                    state_changed = not np.array_equal(current_grid, next_state)
                    
                    done            = (step == n - 1)
                    # Correct step_distance_to_target: 1 for last step, 2 for penultimate, etc.
                    step_distance   = n - step
                    info            = {"step_distance_to_target": step_distance, "transition_type": "random"}
                    transition      = create_transition_dict(
                        state=current_grid,
                        target_state=target_state,
                        action=action,
                        action_space=action_space,
                        reward=reward,
                        next_state=next_state,
                        done=done,
                        info=info,
                        canvas_size=canvas_size,
                    )
                    buffer.append(transition)
                    transitions_added += 1
                    
                    if transitions_added % 10 == 0 or transitions_added == buffer_size:
                        print(f"[Random] Added {transitions_added}/{buffer_size} transitions to buffer.")
                    current_grid = next_state.copy()
                    
                    # Only increment step counter if state changed or if skip_no_change_steps is False
                    if state_changed or not skip_no_change_steps:
                        step += 1
                except Exception:
                    # If action fails, skip it and continue
                    continue

    print(f"[Mixed] Buffer generation complete. Total transitions: {len(buffer)}")
    return buffer

def load_config(config_path: str = "buffer/factorized_buffer_config.yaml") -> dict:
    """
    Load YAML config for buffer generation. Raises clear error if missing or malformed.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}\nPlease create it (see buffer/buffer_config.yaml.example)")
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to parse YAML config: {e}")
    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} is not a valid YAML mapping.")
    return config

def save_buffer_to_pt(buffer: List[Dict[str, Any]], filepath: str):
    """
    Saves the replay buffer to a PyTorch .pt file.
    The buffer (a list of dicts) is converted into a dictionary of torch tensors
    for efficient storage.
    """
    if not buffer:
        print("Warning: Buffer is empty, not saving to PT.")
        return

    print(f"Preparing to save buffer to PT at {filepath}...")
    
    # Initialize a dictionary to hold columns of data
    replay_data = {
        'state': [], 'target_state': [], 'next_state': [], 'selection_mask': [],
        'action_colour': [], 'action_selection': [], 'action_transform': [],
        'reward': [], 'done': [], 'colour': [], 
        'shape_h': [], 'shape_w': [], 'num_colors_grid': [], 'most_present_color': [], 'least_present_color': [],
        'shape_h_target': [], 'shape_w_target': [], 'shape_h_next': [], 'shape_w_next': [],
        'num_colors_grid_target': [], 'most_present_color_target': [], 'least_present_color_target': [],
        'transition_type': [], 'step_distance_to_target': [], 'num_colors_grid_next': [], 'most_present_color_next': [], 'least_present_color_next': [],
    }

    # Populate the dictionary from the buffer
    for transition in buffer:
        replay_data['state'].append(transition['state'])
        replay_data['target_state'].append(transition['target_state'])
        replay_data['next_state'].append(transition['next_state'])
        replay_data['selection_mask'].append(transition['selection_mask'])
        
        replay_data['action_colour'].append(transition['action']['colour'])
        replay_data['action_selection'].append(transition['action']['selection'])
        replay_data['action_transform'].append(transition['action']['transform'])
        
        replay_data['reward'].append(transition['reward'])
        replay_data['done'].append(transition['done'])
        replay_data['colour'].append(transition['colour'])

        shape = transition.get('shape', (0, 0))
        replay_data['shape_h'].append(shape[0])
        replay_data['shape_h_target'].append(transition.get('shape_target', (0, 0))[0])
        replay_data['shape_h_next'].append(transition.get('shape_next', (0, 0))[0])
        replay_data['shape_w'].append(shape[1])
        replay_data['shape_w_target'].append(transition.get('shape_target', (0, 0))[1])
        replay_data['shape_w_next'].append(transition.get('shape_next', (0, 0))[1])
        replay_data['num_colors_grid'].append(transition.get('num_colors_grid'))
        replay_data['num_colors_grid_target'].append(transition.get('num_colors_grid_target'))
        replay_data['num_colors_grid_next'].append(transition.get('num_colors_grid_next'))
        replay_data['most_present_color'].append(transition.get('most_present_color'))
        replay_data['most_present_color_target'].append(transition.get('most_present_color_target'))
        replay_data['most_present_color_next'].append(transition.get('most_present_color_next'))
        replay_data['least_present_color'].append(transition.get('least_present_color'))
        replay_data['least_present_color_target'].append(transition.get('least_present_color_target'))
        replay_data['least_present_color_next'].append(transition.get('least_present_color_next'))

        info = transition.get('info', {})
        replay_data['transition_type'].append(info.get('transition_type', ''))
        replay_data['step_distance_to_target'].append(info.get('step_distance_to_target', -1))

    # Convert to torch tensors and save
    tensor_data = {}
    for key, data in replay_data.items():
        if key == 'transition_type':
            # Keep strings as list (torch doesn't handle strings well in tensors)
            tensor_data[key] = data
        else:
            tensor_data[key] = torch.tensor(np.array(data))

    torch.save(tensor_data, filepath)
    print(f"Successfully saved buffer with {len(buffer)} transitions to {filepath}")


def main():
    """
    Main function to generate the ARC replay buffer with factorized action space.
    Now uses buffer/buffer_config.yaml for all configuration.
    """
    config                          = load_config()
    
    # Set defaults for any missing fields (should match buffer_config.yaml)
    defaults                        = {
        'mode': 'mixed',
        'buffer_size': 500000,
        'num_steps_per_grid': 5,
        'canvas_size': 10,
        'seed': None,
        'action_preset': 'default',
        'challenges_json_path': 'data/challenges.json',
        'solutions_json_path': 'data/solutions.json',
        'max_grid_dim_h': 5,
        'max_grid_dim_w': 5,
        'random_grid_h': 5,
        'random_grid_w': 5,
        'random_num_colors': 4,
        'random_lambda_poisson': 7.0,
        'skip_no_change_steps': True,
    }
    
    for k, v in defaults.items():
        if k not in config or config[k] is None:
            config[k]               = v

    # Validate random_num_colors
    if config['random_num_colors'] > 10:
        raise ValueError("random_num_colors must be <= 10 (ARC palette constraint)")

    if config['seed'] is not None:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
    else:
        random.seed(None)
        np.random.seed(None)

    action_space = ARCActionSpace(preset=config['action_preset'], mode="factorized")
    env                             = ARCEnv(
        challenges_json=config['challenges_json_path'],
        solutions_json=config['solutions_json_path'],
        action_space=action_space,
        canvas_size=config['canvas_size'],
        seed=config['seed']
    )
    
    # Load solutions and challenges data
    with open(config['solutions_json_path'], 'r') as f:
        solutions_data = json.load(f)
    with open(config['challenges_json_path'], 'r') as f:
        challenges_data = json.load(f)
    
    if config['mode'] == "challenge":
        valid_initial_grids = []
        for task_key in challenges_data:
            if 'train' in challenges_data[task_key]:
                for i, example in enumerate(challenges_data[task_key]['train']):
                    input_grid = np.array(example['input'])
                    output_grid = np.array(solutions_data[task_key])
                    if (input_grid.shape[0] <= config['max_grid_dim_h'] and input_grid.shape[1] <= config['max_grid_dim_w'] and
                        output_grid.shape[0] <= config['max_grid_dim_h'] and output_grid.shape[1] <= config['max_grid_dim_w']):
                        valid_initial_grids.append((task_key, i))
        if not valid_initial_grids:
            print(f"No initial grids found matching criteria (max_h={config['max_grid_dim_h']}, max_w={config['max_grid_dim_w']}). Exiting.")
            return
        buffer                      = generate_buffer_from_challenges(
            env=env,
            action_space=action_space,
            valid_initial_grids=valid_initial_grids,
            buffer_size=config['buffer_size'],
            num_steps_per_grid=config['num_steps_per_grid'],
            canvas_size=config['canvas_size'],
        )
    
    elif config['mode'] == "random":
        buffer = generate_buffer_from_random(
            env=env,
            action_space=action_space,
            buffer_size=config['buffer_size'],
            grid_shape=(config['random_grid_h_lambda'], config['random_grid_w_lambda']),
            num_colors=config['random_num_colors'],
            lambda_poisson=config['random_lambda_poisson'],
            canvas_size=config['canvas_size'],
        )
    
    elif config['mode'] == "mixed":
        valid_initial_grids = []
        for task_key in challenges_data:
            if 'train' in challenges_data[task_key]:
                for i, example in enumerate(challenges_data[task_key]['train']):
                    input_grid = np.array(example['input'])
                    output_grid = np.array(solutions_data[task_key])
                    if (input_grid.shape[0] <= config['max_grid_dim_h'] and input_grid.shape[1] <= config['max_grid_dim_w'] and
                        output_grid.shape[0] <= config['max_grid_dim_h'] and output_grid.shape[1] <= config['max_grid_dim_w']):
                        valid_initial_grids.append((task_key, i))
        if not valid_initial_grids:
            print(f"No initial grids found matching criteria (max_h={config['max_grid_dim_h']}, max_w={config['max_grid_dim_w']}). Exiting.")
            return
        buffer                      = generate_buffer_mixed(
            env=env,
            valid_initial_grids=valid_initial_grids,
            num_steps_per_grid=config['num_steps_per_grid'],
            action_space=action_space,
            buffer_size=config['buffer_size'],
            grid_shape_lambda=(config['random_grid_h'], config['random_grid_w']),
            num_colors=config['random_num_colors'],
            lambda_poisson=config['random_lambda_poisson'],
            canvas_size=config['canvas_size'],
            skip_no_change_steps=config['skip_no_change_steps'],
        )
    else:
        raise ValueError(f"Invalid mode: {config['mode']}")

    # --- Saving Logic ---
    # Define output directory and ensure it exists
    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct dynamic filename
    filename = f"rb_{config['mode']}{config['buffer_size']}{config['canvas_size']}_{config['action_preset']}.pt"
    filepath = output_dir / filename
    
    # Save the buffer to PyTorch format
    save_buffer_to_pt(buffer, str(filepath))

    # Minimal assertion to check new fields in at least one transition
    if buffer:
        t = buffer[0]
        assert "shape" in t and isinstance(t["shape"], (tuple, list)), "Missing or invalid 'shape' field in transition."
        assert "num_colors_grid" in t and isinstance(t["num_colors_grid"], int), "Missing or invalid 'num_colors_grid' field in transition."
        assert "most_present_color" in t and isinstance(t["most_present_color"], int), "Missing or invalid 'most_present_color' field in transition."
        assert "least_present_color" in t and isinstance(t["least_present_color"], int), "Missing or invalid 'least_present_color' field in transition."

        print(f"Sample transition new fields: shape={t['shape']}, num_colors_grid={t['num_colors_grid']}, most_present_color={t['most_present_color']}")

        # Load and verify the saved data
        saved_data = torch.load(filepath)
        print("Top-level keys:", list(saved_data.keys()))

if __name__ == "__main__":
    main()