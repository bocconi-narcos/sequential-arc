from __future__ import annotations

import json # For example usage, not directly used by core logic here
from typing import List, Type, Dict, Any, Optional, Union # Added Union
from pathlib import Path # For example usage

from .base import BaseConfig
from .environment import EnvironmentConfig
from .action_space import ActionSpaceConfig
from .solver import SolverConfig # , HeuristicSolverConfig, RLSolverConfig # if needed

# This module can be used for:
# 1. Centralized validation functions that might involve multiple config objects.
# 2. Utility functions for validating specific complex fields or conditions.
# 3. Registering and dispatching validation routines.

class ConfigValidationError(ValueError):
    """Custom exception for configuration validation errors."""
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors if errors is not None else []

def validate_master_config(
    env_config: EnvironmentConfig,
    action_config: ActionSpaceConfig,
    solver_config: SolverConfig,
    # Potentially other configs like data_config, logging_config etc.
) -> None:
    """
    Validates a set of related configuration objects.
    This function can check for inconsistencies or dependencies between different
    parts of the configuration.

    Args:
        env_config: The environment configuration.
        action_config: The action space configuration.
        solver_config: The solver configuration.

    Raises:
        ConfigValidationError: If any cross-validation fails.
    """
    errors: List[str] = []

    # Individual validations are already called by __post_init__ of each config.
    # This function is for cross-config checks.

    # Example: Check if action space preset is compatible with something in env_config
    # if action_config.preset == "special_for_small_canvas" and env_config.canvas_size > 10:
    #     errors.append(
    #         f"Action preset '{action_config.preset}' is intended for canvas_size <= 10, "
    #         f"but EnvironmentConfig has canvas_size={env_config.canvas_size}."
    #     )

    # Example: Check if solver type makes sense with action space mode
    # if isinstance(solver_config, RLSolverConfig) and action_config.mode == "joint" and \
    #    solver_config.model_architecture == "factorized_specific_model":
    #     errors.append(
    #         "RL solver with 'factorized_specific_model' architecture might expect "
    #         "a 'factorized' action space, but mode is 'joint'."
    #     )

    # Example: If data_path in EnvironmentConfig is None, some solvers might not work if they require data.
    if env_config.data_path is None:
        if solver_config.solver_type in ["data_driven_heuristic", "rl_pretrained_on_arc_data"]: # Fictional types
            errors.append(
                f"Solver type '{solver_config.solver_type}' typically requires data, "
                "but EnvironmentConfig.data_path is not set."
            )

    if errors:
        error_messages = "\n - ".join(errors)
        raise ConfigValidationError(
            f"Master configuration validation failed with the following issues:\n - {error_messages}",
            errors=errors
        )

    print("Master configuration validation passed (cross-config checks).")


def validate_config_files(config_paths: Dict[str, str], config_types: Dict[str, Type[BaseConfig]]) -> Dict[str, BaseConfig]:
    """
    Loads and validates multiple configuration files.

    Args:
        config_paths: A dictionary mapping config names to file paths.
                      e.g., {"env": "path/to/env.json", "solver": "path/to/solver.json"}
        config_types: A dictionary mapping config names to their respective BaseConfig subclasses.
                      e.g., {"env": EnvironmentConfig, "solver": SolverConfig}

    Returns:
        A dictionary of loaded and validated configuration objects.

    Raises:
        FileNotFoundError: If a config file is not found.
        ConfigValidationError: If validation of any config fails.
        ValueError: If JSON parsing fails or other issues occur.
    """
    loaded_configs: Dict[str, BaseConfig] = {}
    for name, path in config_paths.items():
        if name not in config_types:
            raise ValueError(f"No configuration type specified for '{name}'.")

        config_cls = config_types[name]
        print(f"Loading and validating '{name}' config from '{path}' using type {config_cls.__name__}...")
        try:
            config_instance = config_cls.load_from_json(path) # load_from_json calls validate()
            loaded_configs[name] = config_instance
            print(f"Successfully loaded and validated '{name}' config.")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file for '{name}' not found at '{path}': {e}")
        except (json.JSONDecodeError, ValueError) as e: # Catches BaseConfig's internal ValueErrors too
            raise ConfigValidationError(f"Error loading or validating config '{name}' from '{path}': {e}")

    # Optionally, perform cross-validation if all required configs are present
    # For example, if "env", "action", and "solver" configs are all loaded,
    # you could call validate_master_config here.
    # This depends on which configs are considered a "complete set".

    return loaded_configs


# Example Usage:
# if __name__ == "__main__":
#     # Create dummy config files for testing validate_config_files
#     import json
#     from pathlib import Path

#     temp_dir = Path("_temp_config_test")
#     temp_dir.mkdir(exist_ok=True)

#     env_data = {"canvas_size": 15, "render_mode": "rgb_array"}
#     action_data = {"mode": "joint", "preset": "minimal"}
#     solver_data = {"solver_type": "heuristic_simple", "hyperparameters": {"depth": 3}}

#     with open(temp_dir / "env.json", "w") as f: json.dump(env_data, f)
#     with open(temp_dir / "action.json", "w") as f: json.dump(action_data, f)
#     with open(temp_dir / "solver.json", "w") as f: json.dump(solver_data, f)

#     config_paths_map = {
#         "environment": str(temp_dir / "env.json"),
#         "action_space": str(temp_dir / "action.json"),
#         "solver_main": str(temp_dir / "solver.json")
#     }
#     config_types_map = {
#         "environment": EnvironmentConfig,
#         "action_space": ActionSpaceConfig,
#         "solver_main": SolverConfig
#     }

#     try:
#         print("\n--- Testing validate_config_files ---")
#         loaded = validate_config_files(config_paths_map, config_types_map)
#         print("All specified configs loaded and individually validated.")

#         env_conf = loaded.get("environment")
#         action_conf = loaded.get("action_space")
#         solver_conf = loaded.get("solver_main")

#         if isinstance(env_conf, EnvironmentConfig) and \
#            isinstance(action_conf, ActionSpaceConfig) and \
#            isinstance(solver_conf, SolverConfig):
#             print("\n--- Testing validate_master_config ---")
#             validate_master_config(env_conf, action_conf, solver_conf)
#         else:
#             print("\nSkipping master config validation due to missing or mismatched config types.")

#     except (ConfigValidationError, FileNotFoundError, ValueError) as e:
#         print(f"Error during config validation tests: {e}")
#         if hasattr(e, 'errors') and e.errors:
#             print("Specific errors:")
#             for err in e.errors:
#                 print(f"  - {err}")

#     finally:
#         # Clean up dummy files
#         import shutil
#         shutil.rmtree(temp_dir, ignore_errors=True)

#     # Example of direct master validation with potential error
#     print("\n--- Testing validate_master_config directly (with potential cross-config issue) ---")
#     try:
#         env_conf_problem = EnvironmentConfig(data_path=None) # No data path
#         action_conf_ok = ActionSpaceConfig()
#         # A fictional solver type that requires data
#         solver_conf_problem = SolverConfig(solver_type="rl_pretrained_on_arc_data")

#         validate_master_config(env_conf_problem, action_conf_ok, solver_conf_problem)
#     except ConfigValidationError as e:
#         print(f"Master validation failed as expected: {e.args[0]}")
#         if e.errors:
#              for err_detail in e.errors: print(f"  - {err_detail}")
