from __future__ import annotations

from typing import Dict, List, Any, Type
from abc import ABC, abstractmethod

# TYPE_CHECKING is used for forward references in type hints that would cause circular imports.
# Useful if OperationRegistry or BaseOperation also imported from this file, though not strictly needed here
# as OperationRegistry and BaseOperation are imported normally.
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from arc_env.dsl.core.operation_registry import OperationRegistry
#     from arc_env.dsl.core.base_operations import BaseOperation

from arc_env.dsl.core.operation_registry import OperationRegistry # Actual import
from arc_env.dsl.core.base_operations import BaseOperation     # Actual import


class BasePresetLoader(ABC):
    """
    Abstract base class for preset loaders.
    Preset loaders are responsible for populating an OperationRegistry
    with a specific set of operations and preset configurations.
    """

    def __init__(self, registry: OperationRegistry):
        if not isinstance(registry, OperationRegistry):
            raise TypeError("registry must be an instance of OperationRegistry")
        self.registry = registry

    @abstractmethod
    def load(self) -> None:
        """
        Loads operations and preset configurations into the registry.
        This method should call `self.registry.register_operation()` and
        `self.registry.register_preset()`.
        """
        pass

    def _register_operations(self, operations_map: Dict[str, Dict[str, Type[BaseOperation]]]) -> None:
        """
        Helper to register multiple operations.
        Args:
            operations_map: {"category": {"op_name": OpClass, ...}, ...}
        """
        for category, ops in operations_map.items():
            for name, op_class in ops.items():
                self.registry.register_operation(category, name, op_class, exist_ok=True) # Allow overwrite for presets

    def _register_preset_config(self, preset_name: str, config: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Helper to register a preset configuration.
        Args:
            preset_name: Name of the preset.
            config: The preset configuration dictionary.
        """
        self.registry.register_preset(preset_name, config, exist_ok=True) # Allow overwrite for presets


# This file could also contain:
# - Functions to discover preset loader classes.
# - A default preset configuration structure if presets are defined declaratively.

# Example structure for a declarative preset:
# DEFAULT_PRESET_OPERATIONS = {
#     "color": {
#         "fill_selected_red": (FillOperation, {"color": Colors.RED}), # Assuming FillOperation and Colors enum
#         "fill_selected_blue": (FillOperation, {"color": Colors.BLUE}),
#     },
#     "transform": {
#         "rotate_90_cw": (RotateOperation, {"angle": 90}),
#     }
# }

# def make_preset_config_from_declarative(
#     op_map: Dict[str, Dict[str, Tuple[Type[BaseOperation], Dict[str, Any]]]]
# ) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     Converts a declarative operation map into a preset configuration structure
#     suitable for OperationRegistry.register_preset.
#     It assumes operation names in the declarative map are unique and will be used
#     as the 'name' in the preset config.
#     """
#     preset_config: Dict[str, List[Dict[str, Any]]] = {}
#     for category, operations in op_map.items():
#         preset_config[category] = []
#         for op_decl_name, (op_class, op_kwargs) in operations.items():
#             # Here, we need the actual registered name of the op_class, or we assume
#             # op_decl_name is how it will be registered.
#             # This helper is more for defining the structure of the preset *data*
#             # rather than registering the ops themselves.
#             # Let's assume op_decl_name refers to an already registered operation name.
#             # Or, if op_class is given, the preset loader needs to ensure these are registered first.
#             # This is slightly circular if the preset loader uses this.

#             # A better approach for declarative presets:
#             # The preset config directly uses operation names that are expected to be registered.
#             # e.g., {"name": "fill_red_op_registered_name", "kwargs": {"color": 1}}
#             pass # This helper needs more thought on how it interacts with registration.

#     return preset_config

# For now, BasePresetLoader provides a simple contract.
# Concrete preset files (default.py, minimal.py) will implement this.
