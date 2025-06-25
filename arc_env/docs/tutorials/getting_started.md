# Getting Started with `arc_env`

This tutorial will guide you through the initial setup of the `arc_env` package and demonstrate how to run a basic interaction with an ARC environment.

## 1. Installation

Before you begin, ensure you have Python 3.8 or newer installed.

### Using pip

The recommended way to install `arc_env` is using pip.

**From PyPI (once published):**
```bash
pip install arc-env
```
*(Note: This command assumes the package is published on PyPI with the name `arc-env`.)*

**From source (for development or local version):**
If you have cloned the `arc_env` repository, navigate to the root directory of the project (where `pyproject.toml` and `setup.py` are located) and run:

```bash
# For a regular install from source
pip install .

# For an editable install (recommended for development)
pip install -e .
```

This will install `arc_env` and its core dependencies (`gymnasium`, `numpy`).

### Optional Dependencies

`arc_env` has optional dependencies for development (testing, linting, formatting) and for specific features (e.g., advanced rendering, specific RL frameworks if examples use them).

To install development dependencies:
```bash
pip install .[dev]
# or for editable install
pip install -e .[dev]
```

To install testing dependencies:
```bash
pip install .[test]
# or
pip install -e .[test]
```
Refer to `pyproject.toml` for the full list of optional dependencies.

## 2. Verifying Installation

After installation, you can verify it by trying to import the package in a Python interpreter:

```python
import arc_env
print(f"arc_env version: {arc_env.__version__}") # Assuming top __init__ imports version

import arc_env.arc_env.environments as arc_environments
print(f"ARCEnv available: {'ARCEnv' in dir(arc_environments)}")
```
If these commands run without error, the installation was successful.

## 3. Basic Environment Usage

Here's a simple script demonstrating how to initialize and interact with `ARCEnv`. This is similar to the `examples/basic_usage.py` script.

```python
# basic_env_interaction.py
import gymnasium as gym
from arc_env.environments.arc_env import ARCEnv
from arc_env.config.environment import EnvironmentConfig
from arc_env.data.loaders.arc_loader import ARCFileLoader # Or other loaders
from pathlib import Path
import tempfile
import json
import shutil

def main():
    # For this example, create temporary dummy task data
    temp_data_dir = Path(tempfile.mkdtemp(prefix="arc_getting_started_"))
    task_file = temp_data_dir / "task001.json"
    dummy_task_content = {
        "train": [{"input": [[1,0],[0,1]], "output": [[0,1],[1,0]]}],
        "test": [{"input": [[2,2],[2,2]], "output": [[3,3],[3,3]]}]
    }
    with open(task_file, "w") as f:
        json.dump(dummy_task_content, f)

    try:
        # Configure the environment
        # The data_path tells ARCEnv to use its default ARCFileLoader for this path
        env_config = EnvironmentConfig(data_path=temp_data_dir, render_mode="ansi")

        # Initialize ARCEnv
        # ARCEnv's default constructor will try to set up a default OperationRegistry
        # and load default DSL presets.
        env = ARCEnv(env_config=env_config)
        print("ARCEnv initialized.")

        # Set a specific task (using the name of our dummy task file without .json)
        env.set_task("task001")
        print(f"Task 'task001' set. Info: {env.get_challenge_info()}")

        # Reset the environment for the first (or current) test case of the task
        observation, info = env.reset(seed=42) # Use a seed for reproducibility
        print("\nEnvironment reset. Initial observation received.")
        # print("Initial task_grid (partial):\n", observation["task_grid"][:5,:5]) # Show part of the grid

        # Run a few random steps
        num_steps = 3
        for i in range(num_steps):
            action = env.action_space.sample()  # Get a random action
            print(f"\nStep {i + 1}/{num_steps}, Action: {action}")

            observation, reward, terminated, truncated, info = env.step(action)

            print(f"  Decoded Action: {info.get('action_decoded_str')}")
            # print("  New task_grid (partial):\n", observation["task_grid"][:5,:5])
            print(f"  Reward: {reward:.2f}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")

            env.render() # Render the current state (to console due to "ansi" mode)

            if terminated or truncated:
                print("Episode finished.")
                break

        env.close()
        print("\nExample finished.")

    finally:
        # Clean up temporary data directory
        shutil.rmtree(temp_data_dir)
        print(f"Cleaned up: {temp_data_dir}")

if __name__ == "__main__":
    main()
```

**To run this script:**
1.  Save it as `basic_env_interaction.py` (or similar) in a location where Python can find the `arc_env` package (e.g., in the root of your cloned `arc_env` repo if you used `pip install -e .`).
2.  Execute it: `python basic_env_interaction.py`

You should see output indicating environment initialization, task setting, steps being taken with random actions, and corresponding rewards/states.

## Next Steps

Congratulations! You've successfully set up `arc_env` and run a basic interaction.

From here, you can explore:
*   Other tutorials in this documentation (see [Tutorials Index](./index.md)).
*   The example scripts in the `arc_env/examples/` directory of the source code.
*   The API Reference for detailed information on classes and functions.
*   Implementing your own custom solvers or DSL operations.

Happy abstracting and reasoning!
