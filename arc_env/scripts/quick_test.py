"""
quick_test.py

A script for performing quick, ad-hoc tests of specific components or functionalities
of the arc_env package. This is useful during development for rapidly checking
if a small change works as expected without running the full test suite.

Usage examples:
- Test if a specific ARC task loads correctly.
- Instantiate a particular DSL operation and apply it to a sample grid.
- Check the output of a utility function.
- Run a few steps in ARCEnv with a specific configuration.
"""
import argparse
import numpy as np
import sys
from pathlib import Path

# Adjust path for direct script execution
# SCRIPT_DIR = Path(__file__).resolve().parent
# PACKAGE_ROOT = SCRIPT_DIR.parent
# sys.path.insert(0, str(PACKAGE_ROOT.parent))

try:
    from arc_env.environments.arc_env import ARCEnv
    from arc_env.config.environment import EnvironmentConfig
    from arc_env.data.loaders.arc_loader import ARCFileLoader # Example loader
    from arc_env.dsl.core.operation_registry import OperationRegistry
    from arc_env.dsl.presets.default import DefaultPresetLoader
    # Import other components you might want to test quickly
    from arc_env.dsl.operations.color import FillOperation # Example operation
    from arc_env.tests.fixtures.sample_grids import GRID_EMPTY_3x3, get_sample_grid # Example grid
except ImportError as e:
    print(f"ImportError: {e}. Ensure 'arc_env' is installed or PYTHONPATH is set.")
    sys.exit(1)

def test_task_loading(task_id: str, data_path: str):
    print(f"\n--- Quick Test: Task Loading ---")
    print(f"Attempting to load task '{task_id}' from data path '{data_path}'...")
    try:
        loader = ARCFileLoader(data_directory=Path(data_path))
        if task_id not in loader.list_available_tasks():
            print(f"  ERROR: Task '{task_id}' not found by loader. Available: {loader.list_available_tasks()}")
            return
        task_data = loader.load_task(task_id)
        print(f"  SUCCESS: Task '{task_id}' loaded.")
        print(f"    Train pairs: {len(task_data.train)}") # type: ignore
        print(f"    Test pairs: {len(task_data.test)}")   # type: ignore
        if task_data.train: # type: ignore
            print(f"    First train input shape: {task_data.train[0]['input'].shape}") # type: ignore
    except Exception as e:
        print(f"  FAILURE: Error loading task '{task_id}': {e}")

def test_dsl_operation(op_name: str, op_category: str, op_kwargs: dict, grid_name: str):
    print(f"\n--- Quick Test: DSL Operation ---")
    print(f"Testing operation '{op_name}' (category '{op_category}') with kwargs {op_kwargs} on grid '{grid_name}'.")

    registry = OperationRegistry()
    DefaultPresetLoader(registry).load() # Load default ops

    op_class = registry.get_operation_class(op_category, op_name)
    if not op_class:
        print(f"  ERROR: Operation class for '{op_name}' (cat: '{op_category}') not found in registry.")
        return

    try:
        operation_instance = op_class(**op_kwargs)
    except Exception as e:
        print(f"  ERROR: Failed to instantiate operation '{op_name}' with kwargs {op_kwargs}: {e}")
        return

    sample_grid = get_sample_grid(grid_name)
    if sample_grid is None:
        print(f"  ERROR: Sample grid '{grid_name}' not found in fixtures.")
        return

    print(f"  Applying to grid (shape {sample_grid.shape}):\n{sample_grid}")
    try:
        # Note: Some operations might require a selection_mask. This test doesn't provide one.
        # Selection operations might return the grid unchanged and update a mask elsewhere.
        if hasattr(operation_instance, 'generate_mask') and op_category == "selection":
            new_mask = operation_instance.generate_mask(np.copy(sample_grid)) # type: ignore
            print(f"  SUCCESS: Operation '{op_name}' (selection) generated mask:\n{new_mask.astype(int)}")
        else:
            modified_grid = operation_instance.apply(np.copy(sample_grid)) # type: ignore
            print(f"  SUCCESS: Operation '{op_name}' applied. Resulting grid:\n{modified_grid}")
            if np.array_equal(sample_grid, modified_grid):
                print("  INFO: Grid was not changed by the operation.")
    except Exception as e:
        print(f"  FAILURE: Error applying operation '{op_name}': {e}")


def test_env_run(task_id: str, data_path: str, num_steps: int = 3):
    print(f"\n--- Quick Test: Environment Run ---")
    print(f"Running ARCEnv with task '{task_id}' from '{data_path}' for {num_steps} steps.")

    env_cfg = EnvironmentConfig(data_path=Path(data_path), render_mode="ansi", canvas_size=10) # Small canvas
    try:
        env = ARCEnv(env_config=env_cfg)
        env.set_task(task_id)
        obs, info = env.reset()
        print(f"  Env reset for task '{info.get('task_id')}'. Initial obs task_grid sum: {np.sum(obs['task_grid'])}")

        for i in range(num_steps):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            print(f"  Step {i+1}: Action (decoded) = {info.get('action_decoded_str')}, Reward = {reward:.2f}, Term={term}, Trunc={trunc}")
            # env.render() # ANSI render will print
            if term or trunc: break
        env.close()
        print(f"  SUCCESS: Environment run completed for task '{task_id}'.")
    except Exception as e:
        print(f"  FAILURE: Error during environment run for '{task_id}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Quick test script for arc_env components.")
    parser.add_argument("--test_type", type=str, default="env_run",
                        choices=["task_load", "dsl_op", "env_run"],
                        help="Type of quick test to perform.")
    # Args for task_load and env_run
    parser.add_argument("--task_id", type=str, default="example_task_01", help="Task ID to test.")
    parser.add_argument("--data_path", type=str, default="_temp_quick_test_data/tasks",
                        help="Path to ARC task data directory (will be created with dummy data if default).")
    # Args for dsl_op
    parser.add_argument("--op_name", type=str, default="fill_selected", help="DSL operation name.")
    parser.add_argument("--op_category", type=str, default="color", help="DSL operation category.")
    parser.add_argument("--op_kwargs_json", type=str, default='{"color": 5}',
                        help="JSON string of kwargs for the DSL operation.")
    parser.add_argument("--grid_name", type=str, default="empty_3x3",
                        help="Name of the sample grid fixture to use (e.g., 'empty_3x3').")
    # Args for env_run
    parser.add_argument("--num_steps", type=int, default=3, help="Number of random steps for env_run.")

    args = parser.parse_args()

    # Prepare dummy data path if using default for task_load or env_run
    is_default_data_path = (args.data_path == "_temp_quick_test_data/tasks")
    actual_data_path = Path(args.data_path)

    if is_default_data_path and (args.test_type == "task_load" or args.test_type == "env_run"):
        actual_data_path.mkdir(parents=True, exist_ok=True)
        # Create a dummy task file for the default task_id
        dummy_task_file = actual_data_path / f"{args.task_id}.json"
        if not dummy_task_file.exists():
            import json
            content = {"train": [{"input":[[1]],"output":[[0]]}], "test": [{"input":[[2]],"output":[[3]]}]}
            with open(dummy_task_file, "w") as f: json.dump(content, f)
            print(f"Created dummy task file: {dummy_task_file}")

    if args.test_type == "task_load":
        test_task_loading(task_id=args.task_id, data_path=str(actual_data_path))
    elif args.test_type == "dsl_op":
        import json # For op_kwargs_json
        try:
            op_kwargs = json.loads(args.op_kwargs_json)
            if not isinstance(op_kwargs, dict): raise ValueError("op_kwargs_json must be a JSON object.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing op_kwargs_json: {e}. Using empty dict.")
            op_kwargs = {}
        test_dsl_operation(op_name=args.op_name, op_category=args.op_category,
                           op_kwargs=op_kwargs, grid_name=args.grid_name)
    elif args.test_type == "env_run":
        test_env_run(task_id=args.task_id, data_path=str(actual_data_path), num_steps=args.num_steps)
    else:
        print(f"Unknown test_type: {args.test_type}")

    # Clean up default dummy data path if it was created
    if is_default_data_path and Path("_temp_quick_test_data").exists():
        import shutil
        shutil.rmtree(Path("_temp_quick_test_data"))
        print(f"Cleaned up default temporary data directory: _temp_quick_test_data")

if __name__ == "__main__":
    main()

"""
Example usage from project root (after `pip install -e .`):

# Test loading a specific task (creates dummy data if using default path)
python arc_env/scripts/quick_test.py --test_type task_load --task_id my_task --data_path path/to/your/tasks

# Test a DSL operation (uses sample_grids.py fixtures)
python arc_env/scripts/quick_test.py --test_type dsl_op --op_name fill_selected --op_category color --op_kwargs_json '{"color": 7}' --grid_name diagonal_3x3_color_2

# Test a short run of the environment (creates dummy data if using default path)
python arc_env/scripts/quick_test.py --test_type env_run --task_id some_task_name --num_steps 5 --data_path path/to/your/tasks

# Run with default settings (env_run with default task_id, creates dummy data)
python arc_env/scripts/quick_test.py
"""
