from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .base import BaseConfig

@dataclass
class EnvironmentConfig(BaseConfig):
    """
    Configuration for the ARC environment.

    Attributes:
        canvas_size: The default size of the environment's canvas (grid).
        step_penalty: Penalty for each step taken.
        shape_penalty: Penalty for incorrect shape of the output grid.
        no_change_penalty: Penalty if an action results in no change to the grid.
        trunc_penalty: Penalty applied when an episode is truncated (e.g., max steps reached).
        completion_bonus: Bonus awarded for correctly solving a task.
        max_steps: Maximum number of steps allowed per episode before truncation.
                   Default is 100. If None, no step limit.
        max_branch: Maximum branching factor for certain operations or search strategies
                    (its specific use might depend on environment or solver logic).
        data_path: Optional path to the ARC dataset files. If provided, the environment
                   might use this to load tasks.
        render_mode: The mode for rendering the environment. Common options are
                     "human", "rgb_array", or None. Default is None.
        render_fps: The target frames per second for rendering when in "human" mode.
                    Default is 4.
    """
    canvas_size: int = 30
    step_penalty: float = -0.1  # Penalties are often negative rewards
    shape_penalty: float = -1.0
    no_change_penalty: float = -0.5
    trunc_penalty: float = -10.0 # Penalty for truncation due to max_steps
    completion_bonus: float = 25.0
    max_steps: Optional[int] = 100 # Maximum steps per episode

    max_branch: int = 1 # This seems low, might need adjustment based on use.
                        # Or it might refer to something specific in the original codebase.

    data_path: Optional[Path] = None # Path to ARC dataset (e.g., JSON files)

    # Gymnasium standard render attributes
    render_mode: Optional[str] = None
    render_fps: int = 4


    def validate(self) -> None:
        """Validate configuration parameters."""
        super().validate() # Call base validation

        if self.canvas_size <= 0:
            raise ValueError("canvas_size must be positive")
        if self.data_path and not self.data_path.exists():
            # This validation might be too strict if the path is created later
            # or if it points to a directory that should exist but might not yet.
            # Consider if it should be a warning or checked at time of use.
            # For now, following the original skeleton's intent.
            # It might be better to check if self.data_path.is_dir() if it's a directory.
            print(f"Warning: data_path '{self.data_path}' does not exist. This might be an issue if data loading is required.")
            # raise ValueError(f"Data path does not exist: {self.data_path}") # Original stricter check

        if self.step_penalty > 0:
            print(f"Warning: step_penalty ({self.step_penalty}) is positive. Penalties are typically zero or negative.")
        if self.shape_penalty > 0:
            print(f"Warning: shape_penalty ({self.shape_penalty}) is positive.")
        if self.no_change_penalty > 0:
            print(f"Warning: no_change_penalty ({self.no_change_penalty}) is positive.")
        if self.trunc_penalty > 0:
            print(f"Warning: trunc_penalty ({self.trunc_penalty}) is positive.")
        if self.completion_bonus < 0:
            print(f"Warning: completion_bonus ({self.completion_bonus}) is negative. Bonuses are typically positive.")

        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive if set.")

        if self.max_branch <= 0:
            raise ValueError("max_branch must be positive.")

        if self.render_fps <= 0:
            raise ValueError("render_fps must be positive.")

        if self.render_mode is not None and self.render_mode not in ["human", "rgb_array", "ansi"]: # "ansi" is another common one
             print(f"Warning: Unknown render_mode '{self.render_mode}'. Supported modes may vary by environment.")


# Example Usage:
# if __name__ == "__main__":
#     try:
#         env_config_default = EnvironmentConfig()
#         print(f"Default Env Config: {env_config_default.to_dict()}")
#         env_config_default.validate() # Called in __post_init__

#         env_config_custom = EnvironmentConfig(
#             canvas_size=20,
#             step_penalty=-0.05,
#             completion_bonus=50,
#             data_path=Path("./non_existent_data_for_test"), # This will print a warning
#             max_steps=200
#         )
#         print(f"Custom Env Config: {env_config_custom.to_dict()}")

#         # Example of saving and loading
#         # temp_config_file = Path("temp_env_config.json")
#         # env_config_custom.save_to_json(temp_config_file)
#         # loaded_env_config = EnvironmentConfig.load_from_json(temp_config_file)
#         # assert loaded_env_config == env_config_custom
#         # print(f"Loaded Env Config: {loaded_env_config.to_dict()}")
#         # temp_config_file.unlink()

#         # Example of validation error
#         # invalid_config = EnvironmentConfig(canvas_size=-5)

#     except ValueError as e:
#         print(f"Configuration Error: {e}")
#     except FileNotFoundError as e:
#         print(f"File Error: {e}")
