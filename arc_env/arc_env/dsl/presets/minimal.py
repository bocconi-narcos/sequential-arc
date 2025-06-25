from __future__ import annotations

from typing import Dict, List, Any, Type

from arc_env.dsl.core.operation_registry import OperationRegistry
from arc_env.dsl.core.base_operations import BaseOperation
from .base import BasePresetLoader

# Import only the operation classes needed for the minimal preset
from arc_env.dsl.operations.color import FillOperation
from arc_env.dsl.operations.selection import SelectByColorOperation, SelectAllOperation
from arc_env.dsl.operations.transformation import RotateOperation # Maybe only one rotation angle

class MinimalPresetLoader(BasePresetLoader):
    """
    Loads the 'minimal' preset into the OperationRegistry.
    The minimal preset includes a very restricted set of essential operations,
    useful for simpler agents or testing.
    """
    PRESET_NAME = "minimal"

    def _get_operations_to_register(self) -> Dict[str, Dict[str, Type[BaseOperation]]]:
        """Defines all operation classes to be registered by this preset loader."""
        # Note: These operations might also be registered by DefaultPresetLoader.
        # The registry's register_operation(exist_ok=True) handles this.
        return {
            "color": {
                "fill_selected": FillOperation,
            },
            "selection": {
                "select_by_color": SelectByColorOperation,
                "select_all": SelectAllOperation, # Useful for applying fill to whole grid
            },
            "transform": {
                "rotate_90_cw": RotateOperation, # Giving it a more specific name for the preset
                                               # if it's always 90deg in minimal.
                                               # The class is still RotateOperation.
            }
        }

    def _get_preset_config(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Defines the configuration for the 'minimal' preset.
        """
        config = {
            "color": [
                {"name": "fill_selected", "kwargs": {"color": 1}}, # Fill with color 1
            ],
            "selection": [
                {"name": "select_by_color", "kwargs": {"color": 0}}, # Select color 0
                {"name": "select_all", "kwargs": {}},
            ],
            "transform": [
                # We registered RotateOperation as "rotate_90_cw" above for this loader.
                # So, the name here must match that.
                {"name": "rotate_90_cw", "kwargs": {"angle": 90}},
            ]
        }
        return config

    def load(self) -> None:
        """
        Registers operations and the 'minimal' preset configuration.
        """
        operations_to_register = self._get_operations_to_register()
        # We need to be careful if operation names here (e.g. "rotate_90_cw") are different
        # from names used in other presets for the SAME class (e.g. "rotate" for RotateOperation).
        # The OperationRegistry registers Type[BaseOperation] under a name for a category.
        # If DefaultPresetLoader registered RotateOperation as "rotate", and MinimalPresetLoader
        # tries to register it as "rotate_90_cw", these will be two distinct entries in the
        # registry if we don't handle it.
        # A better way: register the class once with a canonical name, then presets refer to it.
        #
        # Let's adjust: The _get_operations_to_register should use canonical names.
        # The preset config then refers to these canonical names and provides specific kwargs.

        # Corrected approach:
        # 1. Ensure operations are registered with canonical names (e.g., by DefaultPresetLoader or a common loader).
        # 2. This preset loader might only need to define its _get_preset_config,
        #    referencing already registered operations.
        # Or, if it *does* register ops, use canonical names.

        # For simplicity of BasePresetLoader, each loader registers what it needs. `exist_ok=True` handles overlaps.
        # If MinimalPresetLoader uses "rotate_90_cw" as the *registered name* for RotateOperation,
        # then the preset config *must* use "rotate_90_cw".
        # If DefaultPresetLoader registered RotateOperation as "rotate", then "rotate" and "rotate_90_cw"
        # would be two different *registered operations* (aliases) pointing to the same class.

        # Sticking to the current BasePresetLoader design:
        self._register_operations(operations_to_register)

        preset_config_data = self._get_preset_config()
        self._register_preset_config(self.PRESET_NAME, preset_config_data)

        print(f"'{self.PRESET_NAME}' preset loaded with {sum(len(ops) for ops in operations_to_register.values())} operation types "
              f"and configuration for {sum(len(cfgs) for cfgs in preset_config_data.values())} preset operations.")


# Example of direct usage:
# if __name__ == "__main__":
#     registry = OperationRegistry()

#     # Minimal preset might depend on some ops already being registered if it doesn't register them itself.
#     # Or, as implemented, it registers its own set (which might overlap with default).
#     minimal_loader = MinimalPresetLoader(registry)
#     minimal_loader.load()

#     print("\nAvailable operations after loading minimal preset:")
#     # This will show ops registered by MinimalPresetLoader. If Default was also loaded, it shows merged.
#     print(registry.list_available_operations())
#     print("\nAvailable presets:", registry.list_available_presets())

#     try:
#         minimal_transform_ops = registry.get_operations(category="transform", preset_name="minimal")
#         print(f"\nMinimal transform operations from preset '{MinimalPresetLoader.PRESET_NAME}':")
#         for op in minimal_transform_ops:
#             print(f"- {op.to_string()} (Class: {op.__class__.__name__}, Registered Name in Preset: 'rotate_90_cw')")
#             # Note: op.to_string() will use the params from the preset config.
#             # The registered name "rotate_90_cw" is how the preset refers to the RotateOperation class.

#     except Exception as e:
#         print(f"Error retrieving operations from minimal preset: {e}")

#     # To illustrate the registration name point:
#     # If DefaultPresetLoader also ran, it would have registered RotateOperation under "transform"/"rotate".
#     # MinimalPresetLoader registered RotateOperation under "transform"/"rotate_90_cw".
#     # So, registry.list_available_operations("transform") would show both "rotate" and "rotate_90_cw".
#     # Both would point to the RotateOperation class. This is fine.
#     # The preset "minimal" specifically uses the "rotate_90_cw" instance.
#     # The preset "default" would use the "rotate" instance.
#     # This allows different presets to use the same operation class but with different default params
#     # by registering them under different names if needed, or just different configs for the same name.
#     # The current `get_preset_config` uses the same name as registered by `_get_operations_to_register`.
#     # This is consistent within the loader.
#     # If "rotate_90_cw" was just a config name for "rotate" op, then _get_operations_to_register
#     # would register RotateOperation as "rotate", and preset config for minimal would be:
#     # {"name": "rotate", "kwargs": {"angle": 90}}
#     # This seems cleaner. Let's assume canonical names for registration, specific params in preset config.
#     # The current BasePresetLoader design means each loader registers ops.
#     # So, MinimalPresetLoader's "rotate_90_cw" *is* the registration name for RotateOperation within its scope.
#     # This is okay. The ActionSpace would then use "rotate_90_cw" when using the minimal preset.
