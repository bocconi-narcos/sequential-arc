#!/usr/bin/env python3

"""
Visualize each action applied by a solver step by step.
This script helps debug solvers by showing the transformation at each step.
"""

import argparse
import time
from pathlib import Path
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from solvers import SOLVERS, get_actions
from action_space import ARCActionSpace
from env import ARCEnv

# adjust paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHALLENGES_JSON = DATA_DIR / "challenges.json"
SOLUTIONS_JSON = DATA_DIR / "solutions.json"

def build_env(seed: int | None = 0) -> ARCEnv:
    space = ARCActionSpace(mode="factorized", preset="default")
    return ARCEnv(
        CHALLENGES_JSON,
        SOLUTIONS_JSON,
        action_space=space,
        seed=seed,
    )

def visualize_solver(key: str, pair_idx: int, delay: float = 1.0, seed: int = 0):
    """
    Visualize each action of a solver step by step.
    
    Args:
        key: The solver key to visualize
        pair_idx: Which train pair to use (0-based index)
        delay: Delay between steps in seconds
        seed: Random seed for reproducibility
    """
    if key not in SOLVERS:
        raise ValueError(f"No solver found for key: {key}")
    
    # Setup environment and get actions
    env = build_env(seed=seed)
    actions = get_actions(key, env.action_space)
    
    # Initialize environment with specific pair
    obs, info = env.reset(options={"key": key, "index": pair_idx})
    
    print(f"\nSolver: {key}")
    print(f"Train pair: {pair_idx}")
    print(f"Number of actions: {len(actions)}")
    print("\nGenerating visualizations for all steps...")
    
    # Take steps and generate all visualizations
    for i, action in enumerate(actions):
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        
        # Show the state
        print(f"\nStep {i+1}/{len(actions)}")
        print(f"Action: {env.action_space.action_to_str(action)}")
        env.render(mode="human")
        plt.pause(delay)  # Add a small delay between steps
        
        if done:
            print(f"\n✓ Solved in {i+1} steps!")
            break
    
    # Keep the windows open until user closes them
    plt.show()

def main():
    p = argparse.ArgumentParser(description="Visualize solver actions step by step")
    p.add_argument(
        "key",
        help="Which solver to visualize",
    )
    p.add_argument(
        "--pair",
        type=int,
        default=0,
        help="Which train pair to use (0-based index)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between steps in seconds",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    
    args = p.parse_args()
    visualize_solver(args.key, args.pair, args.delay, args.seed)

if __name__ == "__main__":
    main() 