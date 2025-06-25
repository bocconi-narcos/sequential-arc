from __future__ import annotations

from typing import Dict, List, Any, Type, Optional, Union # Added Union
import json
from pathlib import Path

from arc_env.dsl.core.operation_registry import OperationRegistry
from arc_env.dsl.core.base_operations import BaseOperation
from .base import BasePresetLoader
from arc_env.exceptions import ConfigurationError, TypeError as CustomTypeError # Using custom TypeError

class CustomPresetLoader(BasePresetLoader):
    """
    Loads a 'custom' preset defined in an external configuration file.
    """
    PRESET_NAME = "custom"

    def __init__(self, registry: OperationRegistry, config_filepath: Union[str, Path]):
        super().__init__(registry)
        self.config_filepath = Path(config_filepath)
        if not self.config_filepath.exists() or not self.config_filepath.is_file():
            raise ConfigurationError(f"Custom preset config file not found or is not a file: {self.config_filepath}")

        self.custom_config_data: Optional[Dict[str, Any]] = None

    def _load_config_from_file(self) -> Dict[str, Any]:
        """Loads the custom preset definition from the JSON file."""
        try:
            with open(self.config_filepath, 'r') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ConfigurationError("Custom preset config file must contain a JSON object (dictionary).")
            return data
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Error decoding JSON from custom preset file {self.config_filepath}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Could not load custom preset file {self.config_filepath}: {e}")

    def _validate_custom_config(self, config_data: Dict[str, Any]) -> None:
        """
        Validates the structure of the loaded custom preset configuration.
        Expected structure:
        {
            "operations_to_register": { // Optional: for operations not in standard loaders
                "category_name": {
                    "op_name_in_registry": "fully.qualified.ClassName",...
                }, ...
            },
            "preset_config": { // Required: defines the 'custom' preset itself
                "category_name": [
                    {"name": "op_name_in_registry", "kwargs": {...}}, ...
                ], ...
            }
        }
        """
        if "preset_config" not in config_data or not isinstance(config_data["preset_config"], dict):
            raise ConfigurationError("Custom preset file must have a 'preset_config' dictionary.")

        if "operations_to_register" in config_data:
            if not isinstance(config_data["operations_to_register"], dict):
                raise ConfigurationError("'operations_to_register' must be a dictionary if present.")
            for cat, ops in config_data["operations_to_register"].items():
                if not isinstance(ops, dict):
                    raise ConfigurationError(f"Category '{cat}' in 'operations_to_register' must be a dictionary.")
                for op_name, class_path in ops.items():
                    if not isinstance(class_path, str):
                        raise ConfigurationError(
                            f"Class path for '{op_name}' in category '{cat}' must be a string."
                        )

        for cat, op_list in config_data["preset_config"].items():
            if not isinstance(op_list, list):
                raise ConfigurationError(f"Operation list for category '{cat}' in 'preset_config' must be a list.")
            for op_conf in op_list:
                if not isinstance(op_conf, dict) or "name" not in op_conf:
                    raise ConfigurationError(f"Invalid operation config in '{cat}': {op_conf}. Must be dict with 'name'.")


    def _dynamically_load_class(self, class_path: str) -> Type[BaseOperation]:
        """Dynamically loads a class given its fully qualified path."""
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            op_class = getattr(module, class_name)
            if not issubclass(op_class, BaseOperation):
                raise CustomTypeError(f"Class {class_path} is not a subclass of BaseOperation.") # Use CustomTypeError
            return op_class
        except ImportError as e:
            raise ConfigurationError(f"Could not import module for class {class_path}: {e}")
        except AttributeError:
            raise ConfigurationError(f"Could not find class {class_name} in module {module_path}.")
        except Exception as e:
            raise ConfigurationError(f"Error loading class {class_path}: {e}")

    def load(self) -> None:
        """
        Loads custom operations (if any) and the 'custom' preset configuration
        from the specified file.
        """
        if self.custom_config_data is None:
            self.custom_config_data = self._load_config_from_file()
            self._validate_custom_config(self.custom_config_data)

        # Register any operations defined in the custom config file
        if "operations_to_register" in self.custom_config_data:
            custom_ops_to_register: Dict[str, Dict[str, Type[BaseOperation]]] = {}
            for category, ops_map in self.custom_config_data["operations_to_register"].items():
                custom_ops_to_register[category] = {}
                for op_name, class_path_str in ops_map.items():
                    op_class = self._dynamically_load_class(class_path_str)
                    custom_ops_to_register[category][op_name] = op_class
            self._register_operations(custom_ops_to_register)
            print(f"Registered {sum(len(ops) for ops in custom_ops_to_register.values())} custom operation types from file.")

        # Register the 'custom' preset itself using its configuration
        custom_preset_definition = self.custom_config_data["preset_config"]
        self._register_preset_config(self.PRESET_NAME, custom_preset_definition)

        print(f"'{self.PRESET_NAME}' preset loaded from '{self.config_filepath}' "
              f"with configuration for {sum(len(cfgs) for cfgs in custom_preset_definition.values())} preset operations.")


# Example of how CustomPresetLoader might be used:
# if __name__ == "__main__":
#     registry = OperationRegistry()

#     # Create a dummy custom config file
#     dummy_custom_config = {
#         "operations_to_register": {
#             "custom_cat": {
#                 # Assuming you have a MyCustomOp in a reachable module my_custom_ops.py
#                 # "my_op": "arc_env.dsl.operations.my_custom_ops.MyCustomOp"
#             }
#         },
#         "preset_config": {
#             "color": [
#                 # Assumes 'fill_selected' is already registered (e.g., by DefaultPresetLoader)
#                 # or defined in "operations_to_register" of this file.
#                 {"name": "fill_selected", "kwargs": {"color": 8}},
#             ],
#             # "custom_cat": [
#             #     {"name": "my_op", "kwargs": {"custom_param": "value"}}
#             # ]
#         }
#     }
#     custom_file = Path("dummy_custom_preset.json")
#     with open(custom_file, "w") as f:
#         json.dump(dummy_custom_config, f, indent=2)

#     # For the example to run, we need 'fill_selected' to be registered.
#     # Let's quickly register a dummy one.
#     from arc_env.dsl.operations.color import FillOperation
#     registry.register_operation("color", "fill_selected", FillOperation)

#     try:
#         custom_loader = CustomPresetLoader(registry, custom_file)
#         custom_loader.load()

#         print("\nAvailable operations after loading custom preset:")
#         print(registry.list_available_operations())
#         print("\nAvailable presets:", registry.list_available_presets())

#         if "custom" in registry.list_available_presets():
#             custom_color_ops = registry.get_operations(category="color", preset_name="custom")
#             print(f"\nCustom color operations from preset 'custom':")
#             for op in custom_color_ops:
#                 print(f"- {op.to_string()}")

#     except ConfigurationError as e:
#         print(f"Configuration Error with custom preset: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#     finally:
#         if custom_file.exists():
#             custom_file.unlink()

# This CustomPresetLoader provides a way to define presets and even new operations
# via external JSON files, making the DSL system more flexible.
# The dynamic class loading part is powerful but requires careful path management
# and security considerations if class paths are user-supplied from untrusted sources.
# For internal use, it's a good way to extend the system without code changes.
