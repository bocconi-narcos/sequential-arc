"""
Utility to inspect a replay buffer pickle file for ARC RL training.

Usage:
    python dsl/utils/inspect_buffer.py --buffer_path path/to/buffer.pkl --num 3 [--verbose] [--plot]
"""
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from action_space import ARCActionSpace


def plot_grids(state, next_state, action, action_space, idx):
    """
    Plot state and next_state side by side with the action name as the title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    cmap = plt.get_cmap('tab10', 10)
    axes[0].imshow(state, cmap=cmap, vmin=0, vmax=9)
    axes[0].set_title('State')
    axes[0].axis('off')
    axes[1].imshow(next_state, cmap=cmap, vmin=0, vmax=9)
    axes[1].set_title('Next State')
    axes[1].axis('off')
    action_str = action_space.action_to_str(action)
    fig.suptitle(f"Transition {idx+1}: {action_str}", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def inspect_buffer(buffer_path, num=3, verbose=False, plot=False, max_colour_selection=False):
    with open(buffer_path, 'rb') as f:
        buffer = pickle.load(f)
    if max_colour_selection:
        max_colour = max(transition['action']['colour'] for transition in buffer)
        print(f"Maximum colour selection in buffer: {max_colour}")
        return
    print(f"Loaded buffer with {len(buffer)} transitions.")
    if not buffer:
        print("Buffer is empty.")
        return
    print(f"Keys in a transition: {list(buffer[0].keys())}")
    action_space = ARCActionSpace(preset="default", mode="factorized")
    for i, transition in enumerate(buffer[:num]):
        print(f"\n--- Transition {i+1} ---")
        state = np.array(transition['state'])
        next_state = np.array(transition['next_state'])
        print(f"state shape: {state.shape}")
        print(f"target_state shape: {np.array(transition['target_state']).shape}")
        print(f"color_in_state: {transition['color_in_state']}")
        print(f"action: {transition['action']}")
        print(f"reward: {transition['reward']}")
        print(f"done: {transition['done']}")
        print(f"info: {transition['info']}")
        print(f"next_state shape: {next_state.shape}")
        if verbose:
            print(f"state array:\n{state}")
            print(f"next_state array:\n{next_state}")
            print(f"selection_mask:\n{np.array(transition['selection_mask'])}")
        if plot:
            plot_grids(state, next_state, transition['action'], action_space, i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a replay buffer pickle file.")
    parser.add_argument('--buffer_path', type=str, required=True, help='Path to the buffer pickle file.')
    parser.add_argument('--num', type=int, default=3, help='Number of transitions to print.')
    parser.add_argument('--verbose', action='store_true', help='Print full state/next_state arrays.')
    parser.add_argument('--plot', action='store_true', help='Plot state and next_state grids for each transition.')
    parser.add_argument('--max-colour-selection', action='store_true', help='Print the maximum colour selection in the buffer and exit.')
    args = parser.parse_args()
    inspect_buffer(args.buffer_path, num=args.num, verbose=args.verbose, plot=args.plot, max_colour_selection=args.max_colour_selection) 