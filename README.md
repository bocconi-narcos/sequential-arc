# ARC with Reinforcement Learning and Sequential Transformations

This project explores solving tasks from the Abstraction and Reasoning Corpus (ARC) using Reinforcement Learning (RL). The core idea is to learn sequences of transformations applied to input grids to match target grids.

**Core Components:**

* **ARC Environment (`arc_env/arc_env.py`):** A Gymnasium-compatible environment that presents ARC tasks to an RL agent. See detailed documentation: [docs/environment.md](docs/environment.md).
* **Hierarchical Action Space (`arc_env/action_space.py`):** Defines the complex, structured actions the agent can take, based on a configurable pipeline of colour selection, grid selection, and transformation. See detailed documentation: [docs/action_space.md](docs/action_space.md).
* **Domain-Specific Language (DSL) (`arc_env/dsl/`):** Modules defining the primitive functions for `colour`, `selection`, and `transform` operations.
* **Action Configuration (`arc_env/action_config.py`):** Defines presets (collections) of available DSL functions that constitute the action space for specific experiments.
* **Solvers (`solvers/`):** Hand-crafted or learned action sequences for specific ARC tasks, useful for testing and baseline comparison.

## Project Status

The RL environment (`ARCEnv`) and the structured action space (`ARCActionSpace`) are implemented. The environment supports loading ARC tasks, stepping through actions, calculating rewards based on grid overlap, and rendering. The action space allows for both 'factorized' (Dict) and 'joint' (Discrete) representations. Initial solvers for some tasks are available.

## Installation

*(Placeholder: Add installation instructions here, e.g., cloning the repo, setting up a virtual environment, installing dependencies via `requirements.txt` or `pyproject.toml`)*

```bash
git clone <your-repo-url>
cd <your-repo-name>
# Example using pip and venv
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

### Checking Solvers

The `check_solvers.py` script verifies that the implemented solvers in the `solvers/` directory correctly solve their corresponding ARC tasks by executing their predefined action sequences within the ARCEnv.

#### Basic Usage:

```bash
python check_solvers.py
```

This command runs all available solvers against their tasks and reports the pass/fail status for each.

#### Options:

- `--keys <key1> <key2> ...`  
  Specify which solvers to test (by their filename without the `.py` extension, e.g., `1e0a9b12`). If omitted, all solvers are tested.

- `--render`  
  Display a plot showing the initial state, the final state after applying the solver's actions, and the target state for each task pair. Saved to `solvers/renders/`.

- `--pair <index>`  
  Only evaluate a specific task pair (0-based index) for the selected solvers.

- `--stepwise`  
  Render the state after each action step. Requires `--pair` to be specified. Useful for debugging a sequence.

- `--seed <int>`  
  Set the random number generator seed for reproducibility (default: `0`).

#### Example:

Test only the solver `1e0a9b12` on its third task pair (index 2), render the final comparison plot, and also render each intermediate step:

```bash
python check_solvers.py --keys 1e0a9b12 --pair 2 --render --stepwise
```