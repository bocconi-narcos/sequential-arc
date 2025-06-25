"""
data_validation.py

A utility script to validate ARC task data using ARCTaskValidator.
It can take a path to a data directory (for ARCFileLoader) or specific
challenge/solution JSON files (for ARCCombinedFileLoader) and report
on the validity of the tasks found.
"""
import argparse
from pathlib import Path
import sys
import json # For creating dummy files if needed for testing the script itself

# Adjust path for direct script execution
# SCRIPT_DIR = Path(__file__).resolve().parent
# PACKAGE_ROOT = SCRIPT_DIR.parent
# sys.path.insert(0, str(PACKAGE_ROOT.parent))

try:
    from arc_env.data.loaders.arc_loader import ARCFileLoader, ARCCombinedFileLoader
    from arc_env.data.loaders.base import BaseDataLoader
    from arc_env.data.processors.validation import ARCTaskValidator
    from arc_env.exceptions import DataLoadError
except ImportError as e:
    print(f"ImportError: {e}. Ensure 'arc_env' is installed or PYTHONPATH is set.")
    sys.exit(1)

def validate_dataset(loader: BaseDataLoader, validator: ARCTaskValidator) -> tuple[int, int]:
    """
    Loads all tasks from the loader and validates each one.
    Returns (number_of_valid_tasks, number_of_invalid_tasks).
    """
    print(f"\nValidating tasks from loader: {loader.__class__.__name__}")
    try:
        task_ids = loader.list_available_tasks()
    except Exception as e:
        print(f"  ERROR: Could not list available tasks from loader: {e}")
        return 0, 0 # No tasks to validate if listing fails

    if not task_ids:
        print("  No tasks found by the loader to validate.")
        return 0, 0

    print(f"  Found {len(task_ids)} tasks to validate.")
    valid_count = 0
    invalid_count = 0

    for i, task_id in enumerate(task_ids):
        print(f"  Validating task {i+1}/{len(task_ids)}: '{task_id}' ... ", end="")
        try:
            task_data = loader.load_task(task_id) # load_task itself might do some validation
            is_structurally_valid = validator.process(task_data) # ARCTaskValidator process

            if is_structurally_valid:
                # Loader's own validation (called within load_task of ARC*Loaders)
                # and ARCTaskValidator both passed.
                # The ARCTaskValidator in loader might be different from the one here.
                # This script's ARCTaskValidator provides an independent check with its own config.
                print("PASSED (ARCTaskValidator)")
                valid_count += 1
            else:
                print("FAILED (ARCTaskValidator)")
                invalid_count += 1
                print(f"    Validation errors for '{task_id}':")
                for err_msg in validator.get_errors():
                    print(f"      - {err_msg}")

        except DataLoadError as dle: # Errors during loading (e.g., JSON parsing, file IO)
            print(f"FAILED (DataLoadError)")
            invalid_count += 1
            print(f"    Loading error for '{task_id}': {dle}")
        except Exception as e: # Other unexpected errors
            print(f"FAILED (Unexpected Error)")
            invalid_count += 1
            print(f"    Unexpected error for '{task_id}': {e}")
            # import traceback; traceback.print_exc() # For debugging script

    print(f"\nValidation Summary for {loader.__class__.__name__}:")
    print(f"  Total tasks processed: {len(task_ids)}")
    print(f"  Valid tasks: {valid_count}")
    print(f"  Invalid tasks: {invalid_count}")
    return valid_count, invalid_count

def main():
    parser = argparse.ArgumentParser(description="Validate ARC task data.")
    parser.add_argument(
        "--loader_type",
        type=str,
        default="file",
        choices=["file", "combined"],
        help="Type of data loader to use: 'file' (directory of JSONs) or 'combined' (master challenge/solution files)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data. For 'file' loader, path to directory. For 'combined', path to challenges.json."
    )
    parser.add_argument(
        "--solutions_path",
        type=str,
        default=None,
        help="Path to solutions.json (required if loader_type is 'combined' and solutions are separate)."
    )
    # Arguments for ARCTaskValidator configuration
    parser.add_argument("--min_color", type=int, default=0, help="Min valid color value.")
    parser.add_argument("--max_color", type=int, default=9, help="Max valid color value.")
    parser.add_argument("--allow_empty_grids", action="store_true", help="Allow grids with zero dimensions.")
    parser.add_argument("--no_require_train", action="store_false", dest="require_train", help="Do not require train pairs.")
    parser.add_argument("--no_require_test", action="store_false", dest="require_test", help="Do not require test pairs.")
    parser.set_defaults(require_train=True, require_test=True)

    args = parser.parse_args()

    # Initialize ARCTaskValidator with CLI arguments
    validator = ARCTaskValidator(
        min_colors=args.min_color,
        max_colors=args.max_color,
        allow_empty_grids=args.allow_empty_grids,
        require_train_pairs=args.require_train,
        require_test_pairs=args.require_test
    )

    data_p = Path(args.data_path)
    loader: Optional[BaseDataLoader] = None

    try:
        if args.loader_type == "file":
            if not data_p.is_dir():
                print(f"Error: For 'file' loader, data_path '{data_p}' must be a directory.")
                sys.exit(1)
            loader = ARCFileLoader(data_directory=data_p)
        elif args.loader_type == "combined":
            if not data_p.is_file():
                print(f"Error: For 'combined' loader, data_path '{data_p}' (challenges.json) must be a file.")
                sys.exit(1)
            if not args.solutions_path:
                print(f"Warning: --solutions_path not provided for 'combined' loader. Assuming solutions are part of challenges file or not strictly needed by it.")
                # ARCCombinedFileLoader handles missing solutions_path by printing a warning if it expects one.
                # It needs a path, even if it's non-existent, to decide.
                # Let's make it a dummy path if not provided, so loader can check existence.
                solutions_p = data_p.parent / "dummy_solutions_non_existent.json" # A path that likely won't exist
            else:
                solutions_p = Path(args.solutions_path)
                if not solutions_p.is_file():
                     print(f"Warning: Provided solutions_path '{solutions_p}' is not a file. Loader might fail or ignore.")
            loader = ARCCombinedFileLoader(challenges_path=data_p, solutions_path=solutions_p)
        else:
            print(f"Error: Unknown loader_type '{args.loader_type}'.")
            sys.exit(1)

    except DataLoadError as dle_init:
        print(f"Error initializing data loader: {dle_init}")
        sys.exit(1)
    except Exception as e_init:
        print(f"Unexpected error initializing data loader: {e_init}")
        sys.exit(1)

    if loader:
        validate_dataset(loader, validator)
    else: # Should be caught by earlier checks
        print("Failed to initialize a valid data loader.")

if __name__ == "__main__":
    # Example: Create dummy files to test the script itself
    # python arc_env/scripts/data_validation.py --loader_type file --data_path _temp_validation_data/file_tasks
    # python arc_env/scripts/data_validation.py --loader_type combined --data_path _temp_validation_data/combined/c.json --solutions_path _temp_validation_data/combined/s.json

    # For a quick self-test of the script logic:
    # setup_dummy_data_for_script_test = False # Set to True to run self-test
    # if setup_dummy_data_for_script_test:
    #     print("--- Running script self-test with dummy data ---")
    #     temp_base = Path("_temp_script_validation_data")
    #     temp_base.mkdir(exist_ok=True)

    #     # File loader dummy data
    #     file_task_dir = temp_base / "file_loader_tasks"
    #     file_task_dir.mkdir(exist_ok=True)
    #     with open(file_task_dir / "good_task.json", "w") as f:
    #         json.dump({"train": [{"input":[[0]], "output":[[1]]}], "test": [{"input":[[2]], "output":[[3]]}]}, f)
    #     with open(file_task_dir / "bad_task_no_test.json", "w") as f:
    #         json.dump({"train": [{"input":[[0]], "output":[[1]]}]}, f) # Missing "test"

    #     sys.argv = ['', '--loader_type', 'file', '--data_path', str(file_task_dir)]
    #     print(f"\nSelf-testing with: {' '.join(sys.argv)}")
    #     main()

    #     # Combined loader dummy data
    #     combined_dir = temp_base / "combined_loader_data"
    #     combined_dir.mkdir(exist_ok=True)
    #     challenges_c = combined_dir / "ch.json"
    #     solutions_c = combined_dir / "sol.json"
    #     with open(challenges_c, "w") as f: json.dump({"taskA": {"train":[], "test":[{"input":[[1]]}]}}, f)
    #     with open(solutions_c, "w") as f: json.dump({"taskA": [[[0]]]}, f)

    #     sys.argv = ['', '--loader_type', 'combined', '--data_path', str(challenges_c), '--solutions_path', str(solutions_c)]
    #     print(f"\nSelf-testing with: {' '.join(sys.argv)}")
    #     main()

    #     import shutil
    #     shutil.rmtree(temp_base)
    #     print(f"Cleaned up self-test dummy data: {temp_base}")
    # else:
    #     main() # Normal execution with CLI args from user
    main()
"""
Example usage from project root (after `pip install -e .`):

# Validate tasks in a directory using ARCFileLoader
python arc_env/scripts/data_validation.py --loader_type file --data_path path/to/arc_task_directory

# Validate tasks from combined JSON files
python arc_env/scripts/data_validation.py --loader_type combined --data_path path/to/challenges.json --solutions_path path/to/solutions.json

# Validate with stricter color range (e.g., only 0-5 allowed)
python arc_env/scripts/data_validation.py --loader_type file --data_path path/to/tasks --max_color 5

# Validate allowing empty grids
python arc_env/scripts/data_validation.py --loader_type file --data_path path/to/tasks --allow_empty_grids
"""
