# All settings are loaded from that file. See the example in the repo for details.
#
# Usage: python rb_generator_factorized.py

import os
import numpy as np
import random
from typing import List, Dict, Any
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
from dsl.utils.random_grid import generate_random_grid

from buffer.utils import (count_unique_colors, validate_colors, count_unique_colors_exclude_padding, 
                     most_present_color_exclude_padding, least_present_color_exclude_padding, get_grid_shape,
                     validate_grid_padding)


def create_transition_dict(
    state: np.ndarray,
    target_state: np.ndarray,
    color: int,
    selection_mask: np.ndarray,
    action: Dict[str, int],
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
    selection_mask_padded           = pad_grid(selection_mask.astype(np.int8), (canvas_size, canvas_size))

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

        "colour": int(color),
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

def generate_buffer_mixed(
    env: ARCEnv,
    action_space: ARCActionSpace,
    buffer_size: int,
    num_steps_per_grid: int,
    canvas_size: int,
    grid_shape_lambda: tuple,
    num_colors: int,
    lambda_poisson: float = 7.0,
    probability_of_solving = 0.5,
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
       
        # --- Random episode (like generate_buffer_from_random) ---
        # 1. Generate initial random grid
        num_rows                = np.clip(np.random.poisson(row_lambda) + 2, 2, 10)
        num_cols                = np.clip(np.random.poisson(col_lambda) + 2, 2, 10)
        grid_shape              = (num_rows, num_cols)

        num_colors              = np.clip(np.random.poisson(3) + 2, 2, 10)
        state                   = generate_random_grid(grid_shape, num_colors)
        n                       = np.random.poisson(lambda_poisson)
        n                       = max(1, n)
        probability_of_changing_action = 1 - probability_of_solving ** (1/n)

        # 2. Generate the sequence of actions that will lead to the target state
        action_sequence_to_target         = []
        temp_grid               = state.copy()
        for _ in range(n):
            grid_collapsed = True
            while grid_collapsed:
                action              = action_space.sample()
                colour_fn, selection_fn, transform_fn = action_space.decode(action)
                colour, selection_mask, new_temp_grid = env.apply_action(
                    colour_fn, selection_fn, transform_fn, temp_grid
                )
                # If all cells are not of the same color, we can use this action
                num_unique_colors = count_unique_colors(new_temp_grid)
                if num_unique_colors > 1:
                    grid_collapsed = False
                    temp_grid = new_temp_grid

            action_sequence_to_target.append(action)
        
        action_sequence       = []
        color_sequence        = []
        mask_sequence         = []
        next_state_sequence   = []
        temp_grid               = state.copy()
        one_action_changed = False
        for i in range(n):
            if np.random.rand() < probability_of_changing_action:
                action              = action_space.sample()
                one_action_changed = True
            elif one_action_changed:
                action = action_space.sample()
            else:
                action = action_sequence_to_target[i]

            colour_fn, selection_fn, transform_fn = action_space.decode(action)
            colour, selection_mask, temp_grid = env.apply_action(
                colour_fn, selection_fn, 
                transform_fn, temp_grid
            )
            action_sequence.append(action)
            color_sequence.append(colour)
            mask_sequence.append(selection_mask)
            next_state_sequence.append(temp_grid.copy())

        target_state            = temp_grid.copy()

        padded_target_state = pad_grid(target_state, (canvas_size, canvas_size))
        padded_start_state = pad_grid(state, (canvas_size, canvas_size))
        env_reset_state = np.stack([padded_start_state, padded_target_state], axis=-1)

        # Reset the environment with the initial state
        state, info = env.external_reset(env_reset_state)
        
        # 3. Roll out the episode using the exact same action sequence, storing transitions
        current_state            = state[..., 0].copy()  # Get the initial grid from the reset state
        action_idx              = 0
        step                    = 0

        while step < n and action_idx < len(action_sequence):
            if len(buffer) >= buffer_size:
                break

            # Use the pre-generated action from the sequence
            action = action_sequence[action_idx]

            # Apply the action to the current grid
            next_state, reward, terminated, truncated, _info = env.step(action)
            env.render()
            done = terminated or truncated
            
            # Check if the state changed
            next_state_unpadded = unpad_grid(next_state[..., 0])
            assert np.all(next_state_unpadded == next_state_sequence[action_idx])

            current_state_unpadded = unpad_grid(current_state)
            target_state_unpadded = unpad_grid(target_state)

            current_color = color_sequence[action_idx]
            current_mask = mask_sequence[action_idx]
            
            # Correct step_distance_to_target: 1 for last step, 2 for penultimate, etc.
            step_distance   = n - step
            info            = {"step_distance_to_target": step_distance, "transition_type": "random"}
            transition      = create_transition_dict(
                state=current_state_unpadded,
                target_state=target_state_unpadded,
                color=current_color,
                selection_mask=current_mask,
                action=action,
                reward=reward,
                next_state= next_state_unpadded,
                done=done,
                info=info,
                canvas_size=canvas_size,
            )
            buffer.append(transition)
            transitions_added += 1
            
            if transitions_added % 10 == 0 or transitions_added == buffer_size:
                print(f"[Random] Added {transitions_added}/{buffer_size} transitions to buffer. \r", end="")
            current_state = next_state[...,0].copy()

            if done:
                break
            
            action_idx += 1

    return buffer

def load_config(config_path: str = "buffer/buffer_config.yaml") -> dict:
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
    env    = ARCEnv(
        challenges_json=config['challenges_json_path'],
        solutions_json=config['solutions_json_path'],
        action_space=action_space,
        canvas_size=config['canvas_size'],
        seed=config['seed']
    )
    
    buffer                      = generate_buffer_mixed(
        env=env,
        num_steps_per_grid=config['num_steps_per_grid'],
        action_space=action_space,
        buffer_size=config['buffer_size'],
        grid_shape_lambda=(config['random_grid_h_lambda'], config['random_grid_w_lambda']),
        num_colors=config['random_num_colors'],
        lambda_poisson=config['random_lambda_poisson'],
        canvas_size=config['canvas_size'],
    )

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