# Creating Custom Solvers

This tutorial will guide you through the process of creating custom solvers for ARC tasks.

## 1. Create a Solver File

First, you need to create a Python file in the `solvers/` directory. The name of the file should be the same as the ARC task key you want to solve (e.g., `1e0a9b12.py`).

## 2. Define the Action Sequence

Inside the solver file, you need to define a variable called `ACTION_SEQUENCE`. This variable should be a list of dictionaries, where each dictionary represents an action.

```python
ACTION_SEQUENCE = [
    {"colour": "most_common", "selection": "all_cells", "transform": "flip_horizontal"},
    {"colour": "least_common", "selection": "all_cells", "transform": "flip_vertical"},
]
```

## 3. Test the Solver

Once you have defined the action sequence, you can test your solver using the `check_solvers.py` script.

```bash
python check_solvers.py --keys 1e0a9b12
```

This will run your solver against the corresponding ARC task and report whether it was successful.
