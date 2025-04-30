# ARC with Reinforcement Learning and Sequential Transformations

This project aims to solve tasks from the Abstraction and Reasoning Corpus (ARC) using Reinforcement Learning (RL) by applying sequential transformations.

## Project Structure

-   **RL Environment:** The core reinforcement learning environment is set up to interact with ARC tasks. It utilizes a specific action space for decision-making.
-   **Action Space (`action_space.py`):** Defines the set of possible actions the RL agent can take.
    -   The specific actions available are configured through `action_config`.
    -   Each action is a composite of three sub-actions:
        1.  **Colour Selection:** Choosing a colour (logic in `colour.py`).
        2.  **Selection:** Selecting pixels/objects based on the chosen colour (logic in `selection.py`).
        3.  **Transformation:** Applying a transformation to the selected elements (logic in `transformation.py`).
-   **Solvers (`solvers/`):** This directory contains specific solvers or approaches developed for individual ARC tasks, potentially serving as baselines or components.

## Current Status

The basic RL environment is implemented, capable of selecting actions defined in `action_space.py` based on the configuration in `action_config`. The composite action structure (colour -> selection -> transformation) is established. The `solvers` directory contains initial solutions for some tasks.

## Usage

### Checking Solvers

The `check_solvers.py` script verifies that the implemented solvers in the `solvers/` directory correctly solve their corresponding ARC tasks.

**Basic Usage:**

```bash
python check_solvers.py
```

This command runs all available solvers against their tasks and reports the pass/fail status for each.

**Options:**

-   `--keys <key1> <key2> ...`: Specify which solvers to test (by their filename without the `.py` extension, e.g., `1e0a9b12`). If omitted, all solvers are tested.
-   `--render`: Display a plot showing the initial state, the final state after applying the solver's actions, and the target state for each task pair.
-   `--pair <index>`: Only evaluate a specific task pair (0-based index) for the selected solvers.
-   `--stepwise`: Render the state after each action step. Requires `--pair` to be specified.

**Example:**

Test only the solver `1e0a9b12` on its third task pair (index 2) and render the result:

```bash
python check_solvers.py --keys 1e0a9b12 --pair 2 --render
```