from __future__ import annotations # For Type[BaseOperation] forward ref if BaseOperation was in same file or complex type hints

from typing import Dict, List, Callable, Type, Any, Optional, Union # Callable not directly used in signatures but good for type hints
from collections import defaultdict

from .base_operations import BaseOperation
from arc_env.exceptions import OperationError, TypeError as CustomTypeError # Using custom TypeError for consistency

class OperationRegistry:
    """
    Central registry for DSL operations and presets.

    This class allows for dynamic registration of operation types and
    predefined sets of operations (presets) that can be used to configure
    action spaces or DSL interpreters.
    """

    def __init__(self) -> None:
        # Stores operation classes: _operations[category][name] = OperationClass
        self._operations: Dict[str, Dict[str, Type[BaseOperation]]] = defaultdict(dict)

        # Stores preset configurations: _presets[preset_name][category] = [op_config, ...]
        # op_config is a dict like {"name": "op_name", "kwargs": {...}}
        self._presets: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    def register_operation(
        self,
        category: str,
        name: str,
        operation_class: Type[BaseOperation],
        exist_ok: bool = False
    ) -> None:
        """
        Register a new operation class.

        Args:
            category: The category of the operation (e.g., "color", "selection", "transform").
            name: The unique name of the operation within its category.
            operation_class: The class (subclass of BaseOperation) implementing the operation.
            exist_ok: If True, re-registering an existing operation with the same name and category
                      will not raise an error (it will be overwritten). Default is False.

        Raises:
            OperationError: If the category is invalid (though currently all are accepted),
                            or if an operation with the same name already exists in the category
                            and exist_ok is False.
            TypeError: If operation_class is not a subclass of BaseOperation.
        """
        if not issubclass(operation_class, BaseOperation):
            raise TypeError(
                f"Operation class '{operation_class.__name__}' must be a subclass of BaseOperation."
            )

        # Currently, all categories are accepted. Could add validation if there's a fixed set.
        # if category not in self._operations:
        #     # This check is effectively handled by defaultdict, but could be explicit
        #     # if categories were predefined and limited.
        #     # For now, new categories can be created on the fly.
        #     pass

        if not exist_ok and name in self._operations[category]:
            raise OperationError(
                f"Operation '{name}' already registered in category '{category}'. "
                "Set exist_ok=True to overwrite."
            )

        self._operations[category][name] = operation_class
        print(f"Registered operation: Category='{category}', Name='{name}', Class='{operation_class.__name__}'")


    def register_preset(
        self,
        preset_name: str,
        preset_config: Dict[str, List[Dict[str, Any]]],
        exist_ok: bool = False
    ) -> None:
        """
        Register a new preset configuration.

        A preset defines a collection of operations, potentially with specific parameters,
        grouped by category.

        Args:
            preset_name: The unique name for the preset (e.g., "default", "minimal_colors").
            preset_config: A dictionary where keys are categories (e.g., "color") and
                           values are lists of operation configurations. Each operation
                           configuration is a dict: `{"name": "op_name", "kwargs": {...}}`.
                           `kwargs` are passed to the operation's constructor.
            exist_ok: If True, re-registering an existing preset will not raise an error.
                      Default is False.

        Raises:
            OperationError: If the preset name already exists and exist_ok is False,
                            or if any operation in the preset_config is not registered.
        """
        if not exist_ok and preset_name in self._presets:
            raise OperationError(
                f"Preset '{preset_name}' already registered. Set exist_ok=True to overwrite."
            )

        # Validate that all operations in the preset are registered
        for category, op_configs in preset_config.items():
            if category not in self._operations:
                # This implies a category used in preset doesn't even exist in _operations.
                # This could be an error, or it could mean the category will have no ops from the registry.
                # For now, let's be strict: if a category is mentioned, it should be known.
                # However, an empty list of ops for a category is fine.
                 print(f"Warning: Category '{category}' in preset '{preset_name}' is not a known operation category. "
                       "If this is intended, ensure operations are added to it later or it's handled.")

            for op_config in op_configs:
                op_name = op_config.get("name")
                if not op_name:
                    raise OperationError(f"Operation in preset '{preset_name}', category '{category}' is missing a 'name'. Config: {op_config}")

                # Check if the operation exists in the registry for that category
                if op_name not in self._operations.get(category, {}):
                    raise OperationError(
                        f"Operation '{op_name}' (category '{category}') in preset '{preset_name}' "
                        "is not registered. Please register the operation first."
                    )

        self._presets[preset_name] = defaultdict(list, {
            cat: list(ops) for cat, ops in preset_config.items()
        }) # Ensure internal structure is defaultdict(list)
        print(f"Registered preset: Name='{preset_name}'")


    def get_operation_class(self, category: str, name: str) -> Optional[Type[BaseOperation]]:
        """
        Retrieve a registered operation class by its category and name.

        Args:
            category: The category of the operation.
            name: The name of the operation.

        Returns:
            The operation class if found, otherwise None.
        """
        return self._operations.get(category, {}).get(name)

    def get_operations_for_preset(
            self,
            preset_name: str,
            category: Optional[str] = None
        ) -> Union[List[BaseOperation], Dict[str, List[BaseOperation]]]:
        """
        Get instantiated operations for a given preset and optionally a specific category.

        Args:
            preset_name: The name of the preset.
            category: If specified, only operations from this category are returned.
                      If None, operations from all categories in the preset are returned,
                      structured as a dictionary.

        Returns:
            If category is specified: A list of instantiated BaseOperation objects.
            If category is None: A dictionary mapping category names to lists of
                                 instantiated BaseOperation objects.

        Raises:
            OperationError: If the preset is not found, or if an operation within
                            the preset cannot be instantiated (e.g., unregistered class
                            or issues with kwargs).
        """
        if preset_name not in self._presets:
            raise OperationError(f"Preset '{preset_name}' not found.")

        instantiated_ops_by_cat: Dict[str, List[BaseOperation]] = defaultdict(list)

        preset_categories_to_process = [category] if category else self._presets[preset_name].keys()

        for cat_key in preset_categories_to_process:
            if cat_key not in self._presets[preset_name]:
                if category: # If a specific category was requested but not in preset
                    raise OperationError(f"Category '{cat_key}' not found in preset '{preset_name}'.")
                continue # Skip if iterating all and category not in this preset

            op_configs = self._presets[preset_name][cat_key]

            for config in op_configs:
                op_name = config.get("name")
                op_kwargs = config.get("kwargs", {})

                if not op_name:
                    # Should have been caught by register_preset, but good to double check
                    raise OperationError(f"Invalid operation config in preset '{preset_name}', category '{cat_key}': missing 'name'.")

                op_class = self.get_operation_class(cat_key, op_name)
                if not op_class:
                    raise OperationError(
                        f"Operation class for '{op_name}' (category '{cat_key}') not found in registry, "
                        f"but it's listed in preset '{preset_name}'. This indicates an inconsistency."
                    )

                try:
                    instantiated_ops_by_cat[cat_key].append(op_class(**op_kwargs))
                except Exception as e:
                    raise OperationError(
                        f"Failed to instantiate operation '{op_name}' (class {op_class.__name__}) "
                        f"from preset '{preset_name}', category '{cat_key}' with kwargs {op_kwargs}. "
                        f"Error: {e}"
                    )

        if category: # Return list for specific category
            return instantiated_ops_by_cat.get(category, [])
        else: # Return dict for all categories
            return dict(instantiated_ops_by_cat)


    def list_available_presets(self) -> List[str]:
        """Returns a list of names of all registered presets."""
        return list(self._presets.keys())

    def list_available_operations(self, category: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
        """
        Lists available operation names, optionally filtered by category.

        Args:
            category: If specified, lists operations only in this category.
                      If None, lists all operations grouped by category.

        Returns:
            If category is specified: A list of operation names in that category.
            If category is None: A dictionary mapping category names to lists of operation names.
        """
        if category:
            return list(self._operations.get(category, {}).keys())
        else:
            return {cat: list(ops.keys()) for cat, ops in self._operations.items() if ops}


# Provided skeleton's get_operations method is similar to get_operations_for_preset.
# The naming in the skeleton was:
# def get_operations(self, category: str, preset: str) -> List[Callable]:
# This implies it always returns a list of callables for a *specific* category and preset.
# My `get_operations_for_preset` is more flexible.
# To match the skeleton's signature, we can add an alias or wrapper:

    def get_operations(self, category: str, preset_name: str) -> List[BaseOperation]:
        """
        Get instantiated operations for a specific category and preset.
        This matches the signature style of the original skeleton.

        Args:
            category: The specific category of operations to retrieve.
            preset_name: The name of the preset.

        Returns:
            A list of instantiated BaseOperation objects for the specified category and preset.

        Raises:
            OperationError: If preset or category within preset is not found, or instantiation fails.
        """
        result = self.get_operations_for_preset(preset_name=preset_name, category=category)
        if isinstance(result, list): # Should be a list if category was specified
            return result
        else:
            # This case should ideally not happen if logic is correct,
            # but as a fallback, return empty list or raise error.
            raise OperationError(f"Unexpected return type from get_operations_for_preset when category '{category}' was specified.")


# Example Usage (for illustration):
# if __name__ == "__main__":
#     class DummyColorOp(BaseOperation):
#         def apply(self, grid, selection_mask=None): return grid
#         def to_string(self): return f"DummyColorOp({self.params})"
#     class DummySelectOp(BaseOperation):
#         def apply(self, grid, selection_mask=None): return grid
#         def to_string(self): return f"DummySelectOp({self.params})"

#     registry = OperationRegistry()
#     registry.register_operation("color", "set_color_red", DummyColorOp)
#     registry.register_operation("color", "set_color_blue", DummyColorOp)
#     registry.register_operation("selection", "select_all", DummySelectOp)

#     print("Registered operations:", registry.list_available_operations())

#     default_preset_config = {
#         "color": [
#             {"name": "set_color_red", "kwargs": {"target_color": 1}},
#             {"name": "set_color_blue", "kwargs": {"target_color": 4}}
#         ],
#         "selection": [
#             {"name": "select_all", "kwargs": {}}
#         ]
#     }
#     registry.register_preset("default", default_preset_config)
#     print("Registered presets:", registry.list_available_presets())

#     try:
#         # Get specific category from preset
#         color_ops_from_default = registry.get_operations("color", "default")
#         print(f"\nColor operations from 'default' preset:")
#         for op in color_ops_from_default:
#             print(f"- {op.to_string()} with params {op.get_params()}")

#         # Get all operations from preset
#         all_ops_from_default = registry.get_operations_for_preset("default")
#         print(f"\nAll operations from 'default' preset:")
#         if isinstance(all_ops_from_default, dict):
#             for cat, ops_list in all_ops_from_default.items():
#                 print(f"  Category: {cat}")
#                 for op in ops_list:
#                     print(f"  - {op.to_string()}")

#         # Example of error: unknown preset
#         # registry.get_operations("color", "unknown_preset")

#     except OperationError as e:
#         print(f"OperationRegistry Error: {e}")
#     except TypeError as e:
#         print(f"Type Error during registration: {e}")
