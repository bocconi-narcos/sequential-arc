from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any

from .base import BaseConfig

@dataclass
class ActionSpaceConfig(BaseConfig):
    """
    Configuration for the ARC action space.

    Attributes:
        mode: The operational mode of the action space.
              - "factorized": Actions are represented as a dictionary of
                discrete choices for different categories (e.g., color, selection, transform).
              - "joint": Actions are represented as a single discrete integer,
                which is then decoded into a combination of operations.
              Default is "factorized".
        preset: Name of the DSL operation preset to use (e.g., "default", "minimal").
                This determines which specific operations are available.
                Default is "default".
        available_presets: A list of known preset names. Used for validation.
                           This might be dynamically populated or extended by a registry.
        custom_operations_config_path: Optional path to a JSON/YAML file defining
                                       custom operations or overriding presets.
        allow_noop: Whether to include a "no operation" action. Default is False.
                    This might be handled by specific operations or as a global option.
    """
    mode: Literal["factorized", "joint"] = "factorized"
    preset: str = "default" # E.g., "default", "minimal", "custom_from_file"

    # This could be dynamically populated by an OperationRegistry in a more advanced setup.
    # For now, we can list some expected ones for validation.
    available_presets: List[str] = field(default_factory=lambda: ["default", "minimal", "custom"])

    custom_operations_config_path: Optional[str] = None # Path to a file for custom ops/presets

    allow_noop: bool = False # Whether a general No-Op action is part of the space

    # Potentially add more granular controls, e.g.:
    # max_selection_points: Optional[int] = None
    # available_colors: Optional[List[int]] = None


    def validate(self) -> None:
        """Validate action space configuration parameters."""
        super().validate()

        if self.mode not in ["factorized", "joint"]:
            raise ValueError(f"Invalid action space mode: {self.mode}. Must be 'factorized' or 'joint'.")

        if self.preset not in self.available_presets and self.preset != "custom": # "custom" implies it might be loaded externally
             # A more robust check would involve querying an actual OperationRegistry
             # to see if the preset is truly available.
            print(f"Warning: Preset '{self.preset}' is not in the predefined available_presets list: {self.available_presets}. "
                  "Ensure this preset is correctly defined or loaded, especially if it's not 'custom'.")

        if self.custom_operations_config_path:
            # Basic check, could be expanded to see if path is valid, file exists, etc.
            # For now, just ensuring it's a string if provided.
            if not isinstance(self.custom_operations_config_path, str):
                raise ValueError("custom_operations_config_path must be a string (filepath).")
            # If preset is not 'custom' but a custom_operations_config_path is given,
            # it might indicate a misconfiguration or an intent to augment a standard preset.
            # The exact behavior would depend on how OperationRegistry handles this.
            if self.preset != "custom":
                print(f"Warning: custom_operations_config_path ('{self.custom_operations_config_path}') is specified, "
                      f"but preset is '{self.preset}' (not 'custom'). The loading mechanism will determine precedence.")


# Example Usage:
# if __name__ == "__main__":
#     try:
#         as_config_default = ActionSpaceConfig()
#         print(f"Default Action Space Config: {as_config_default.to_dict()}")

#         as_config_joint = ActionSpaceConfig(mode="joint", preset="minimal")
#         print(f"Joint Minimal Action Space Config: {as_config_joint.to_dict()}")

#         as_config_custom = ActionSpaceConfig(
#             preset="custom",
#             custom_operations_config_path="path/to/my_ops.json",
#             available_presets=["default", "minimal", "custom_user_preset_1"] # User can extend this
#         )
#         as_config_custom.available_presets.append("my_loaded_preset") # Simulate dynamic addition
#         print(f"Custom Action Space Config: {as_config_custom.to_dict()}")


#         # Example of validation:
#         # as_config_default.validate() # Called in __post_init__

#         # Trigger warning for preset:
#         # as_config_unknown_preset = ActionSpaceConfig(preset="non_existent_preset")
#         # print(f"Unknown preset config: {as_config_unknown_preset.to_dict()}")


#         # Trigger error for mode:
#         # invalid_mode_config = ActionSpaceConfig(mode="something_else")

#     except ValueError as e:
#         print(f"Configuration Error: {e}")
