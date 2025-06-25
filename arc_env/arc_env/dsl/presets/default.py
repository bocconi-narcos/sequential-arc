from __future__ import annotations

from typing import Dict, List, Any, Type

from arc_env.dsl.core.operation_registry import OperationRegistry
from arc_env.dsl.core.base_operations import BaseOperation
from .base import BasePresetLoader

# Import all operation classes that will be part of the default preset
# These imports assume the operation files (color.py, selection.py etc.) are structured correctly.
from arc_env.dsl.operations.color import (
    ChangeColorOperation,
    FillOperation,
    # Import others like FloodFillOperation if defined and used
)
from arc_env.dsl.operations.selection import (
    SelectByColorOperation,
    SelectAllOperation,
    ClearSelectionOperation,
    InvertSelectionOperation,
    # Import others
)
from arc_env.dsl.operations.transformation import (
    RotateOperation,
    FlipOperation,
    # Import others
)
from arc_env.dsl.operations.composition import (
    CopyPasteOperation,
    OverlayOperation,
    # Import others
)

class DefaultPresetLoader(BasePresetLoader):
    """
    Loads the 'default' preset into the OperationRegistry.
    The default preset includes a comprehensive set of commonly used operations.
    """

    PRESET_NAME = "default"

    def _get_operations_to_register(self) -> Dict[str, Dict[str, Type[BaseOperation]]]:
        """Defines all operation classes to be registered by this preset loader."""
        return {
            "color": {
                "change_color": ChangeColorOperation,
                "fill_selected": FillOperation,
                # "flood_fill": FloodFillOperation, # Example
            },
            "selection": {
                "select_by_color": SelectByColorOperation,
                "select_all": SelectAllOperation,
                "clear_selection": ClearSelectionOperation,
                "invert_selection": InvertSelectionOperation,
            },
            "transform": {
                "rotate": RotateOperation,
                "flip": FlipOperation,
            },
            "composition": {
                "copy_paste": CopyPasteOperation,
                "overlay_selection": OverlayOperation,
            }
            # Add other categories and their operations
        }

    def _get_preset_config(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Defines the configuration for the 'default' preset.
        This specifies which registered operations are part of this preset
        and with what default parameters (if any).
        """
        # These kwargs are examples. Actual useful default parameters would depend
        # on how the operations are designed and typically used.
        # Often, presets might list operations without specific kwargs, meaning
        # the operation will be available with its own default internal parameters
        # or require parameters to be fully specified at runtime if not all are optional.
        config = {
            "color": [
                {"name": "change_color", "kwargs": {"from_color": 0, "to_color": 1}}, # Example default
                {"name": "fill_selected", "kwargs": {"color": 1}}, # Example default
                # {"name": "flood_fill", "kwargs": {"start_point": (0,0), "target_color":0, "replacement_color":1}},
            ],
            "selection": [
                {"name": "select_by_color", "kwargs": {"color": 1}}, # Example
                {"name": "select_all", "kwargs": {}},
                {"name": "clear_selection", "kwargs": {}},
                {"name": "invert_selection", "kwargs": {}},
            ],
            "transform": [
                {"name": "rotate", "kwargs": {"angle": 90}}, # Example
                {"name": "flip", "kwargs": {"axis": "horizontal"}}, # Example
            ],
            "composition": [
                {"name": "copy_paste", "kwargs": {"target_row": 0, "target_col": 0, "ignore_color": None}},
                {"name": "overlay_selection", "kwargs": {"transparent_color": 0}},
            ]
        }
        return config

    def load(self) -> None:
        """
        Registers all operations defined in _get_operations_to_register()
        and then registers the preset configuration from _get_preset_config().
        """
        operations_to_register = self._get_operations_to_register()
        self._register_operations(operations_to_register)

        preset_config_data = self._get_preset_config()
        self._register_preset_config(self.PRESET_NAME, preset_config_data)

        print(f"'{self.PRESET_NAME}' preset loaded with {sum(len(ops) for ops in operations_to_register.values())} operation types "
              f"and configuration for {sum(len(cfgs) for cfgs in preset_config_data.values())} preset operations.")

# To make this preset discoverable or automatically loaded, you might have a manager:
# preset_loader_classes = [DefaultPresetLoader]
# def load_all_presets(registry: OperationRegistry):
#     for loader_class in preset_loader_classes:
#         loader = loader_class(registry)
#         loader.load()

# Example of direct usage:
# if __name__ == "__main__":
#     registry = OperationRegistry()
#     default_loader = DefaultPresetLoader(registry)
#     default_loader.load()

#     print("\nAvailable operations after loading default preset:")
#     print(registry.list_available_operations())
#     print("\nAvailable presets:", registry.list_available_presets())

#     try:
#         default_color_ops = registry.get_operations(category="color", preset_name="default")
#         print(f"\nDefault color operations from preset '{DefaultPresetLoader.PRESET_NAME}':")
#         for op in default_color_ops:
#             print(f"- {op.to_string()}")

#         all_default_ops = registry.get_operations_for_preset(preset_name="default")
#         if isinstance(all_default_ops, dict):
#             print(f"\nAll operations in '{DefaultPresetLoader.PRESET_NAME}' preset by category:")
#             for cat, op_list in all_default_ops.items():
#                 print(f"  Category: {cat}")
#                 for op_instance in op_list:
#                     print(f"    - {op_instance.to_string()}")
#     except Exception as e:
#         print(f"Error retrieving operations from preset: {e}")
