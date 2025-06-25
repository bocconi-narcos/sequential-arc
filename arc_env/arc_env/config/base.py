from __future__ import annotations

from dataclasses import dataclass, asdict, fields, is_dataclass
from typing import Dict, Any, TypeVar, Type, Union # Added Union
import json
from pathlib import Path

T = TypeVar('T', bound='BaseConfig')

@dataclass
class BaseConfig:
    """
    Base configuration class with validation and serialization.

    All configuration classes should inherit from this base class.
    It provides basic methods for validation, dictionary conversion,
    and loading/saving from/to JSON files.
    """

    def validate(self) -> None:
        """
        Validate configuration parameters.
        Subclasses should override this method to implement specific
        validation logic. This method should raise ValueError or a
        custom configuration error if validation fails.
        """
        # Basic validation: check if all fields have been initialized (though dataclasses usually handle this)
        for f in fields(self):
            if not hasattr(self, f.name):
                # This case is unlikely with dataclasses unless __post_init__ deletes an attribute
                # or if a non-default field is not provided during instantiation.
                raise ValueError(f"Configuration field '{f.name}' is missing.")
        pass # Default implementation does nothing.

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration object to a dictionary.
        Handles nested BaseConfig objects and Path objects.
        """
        data = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, BaseConfig):
                data[f.name] = value.to_dict()
            elif isinstance(value, Path):
                data[f.name] = str(value) # Serialize Path objects to strings
            elif is_dataclass(value) and not isinstance(value, BaseConfig): # Other dataclasses
                data[f.name] = asdict(value)
            elif isinstance(value, list):
                data[f.name] = [
                    item.to_dict() if isinstance(item, BaseConfig) else
                    str(item) if isinstance(item, Path) else
                    asdict(item) if is_dataclass(item) and not isinstance(item, BaseConfig) else
                    item
                    for item in value
                ]
            elif isinstance(value, dict):
                data[f.name] = {
                    k: v.to_dict() if isinstance(v, BaseConfig) else
                       str(v) if isinstance(v, Path) else
                       asdict(v) if is_dataclass(v) and not isinstance(v, BaseConfig) else
                       v
                    for k, v in value.items()
                }
            else:
                data[f.name] = value
        return data

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create a configuration object from a dictionary.
        This is a basic implementation. Subclasses might need to override it
        if they have complex types (like Path objects or nested configs)
        that need special handling during deserialization.
        """
        # A more robust implementation would inspect field types and handle them.
        # For now, we rely on dataclass's __init__ type hints for simple cases.
        # This basic version might not correctly reconstruct nested BaseConfig or Path objects
        # without further logic or if __init__ doesn't expect string paths.

        # Improved from_dict to handle Path and nested BaseConfig objects:
        field_types = {f.name: f.type for f in fields(cls)}
        processed_data = {}

        for key, value in data.items():
            if key not in field_types:
                # Pass through keys not in fields; dataclass init will handle unknown kwargs
                processed_data[key] = value
                continue

            field_type_hint = field_types[key]
            origin_type = getattr(field_type_hint, '__origin__', None)
            type_args = getattr(field_type_hint, '__args__', tuple())

            # Resolve Optional[T] to T
            actual_field_type = field_type_hint
            if origin_type is Union and type(None) in type_args and len(type_args) == 2:
                actual_field_type = next(t for t in type_args if t is not type(None))
                origin_type = getattr(actual_field_type, '__origin__', None) # Update origin for List[Path], etc.
                type_args = getattr(actual_field_type, '__args__', tuple())


            if value is None: # Handle None values directly if field is Optional
                processed_data[key] = None
                continue

            # Handle List[Path], List[BaseConfig], List[OtherDataclass]
            if origin_type is list and type_args:
                item_type = type_args[0]
                if isinstance(value, list):
                    processed_list = []
                    for item_val in value:
                        if hasattr(item_type, 'from_dict') and isinstance(item_val, dict): # BaseConfig or dataclass with from_dict
                            processed_list.append(item_type.from_dict(item_val))
                        elif item_type is Path and isinstance(item_val, str):
                            processed_list.append(Path(item_val))
                        elif is_dataclass(item_type) and not hasattr(item_type, 'from_dict') and isinstance(item_val, dict):
                            processed_list.append(item_type(**item_val)) # Basic dataclass init
                        else:
                            processed_list.append(item_val) # Assume item_val is already correct type or convertible
                    processed_data[key] = processed_list
                else:
                    processed_data[key] = value # Let dataclass init handle type error if list expected

            # Handle Dict[str, Path], Dict[str, BaseConfig], etc.
            elif origin_type is dict and type_args and len(type_args) == 2:
                # Assuming key_type is str for simplicity here, focusing on value_type
                val_type = type_args[1]
                if isinstance(value, dict):
                    processed_dict = {}
                    for k_item, v_item in value.items():
                        if hasattr(val_type, 'from_dict') and isinstance(v_item, dict):
                            processed_dict[k_item] = val_type.from_dict(v_item)
                        elif val_type is Path and isinstance(v_item, str):
                            processed_dict[k_item] = Path(v_item)
                        elif is_dataclass(val_type) and not hasattr(val_type, 'from_dict') and isinstance(v_item, dict):
                            processed_dict[k_item] = val_type(**v_item)
                        else:
                            processed_dict[k_item] = v_item
                    processed_data[key] = processed_dict
                else:
                    processed_data[key] = value

            # Handle single Path, BaseConfig, or other dataclass
            elif hasattr(actual_field_type, 'from_dict') and isinstance(value, dict): # Nested BaseConfig or dataclass with from_dict
                processed_data[key] = actual_field_type.from_dict(value)
            elif actual_field_type is Path and isinstance(value, str): # Path objects
                processed_data[key] = Path(value)
            elif is_dataclass(actual_field_type) and not hasattr(actual_field_type, 'from_dict') and isinstance(value, dict):
                processed_data[key] = actual_field_type(**value) # Basic dataclass init
            else:
                processed_data[key] = value # Fallback, rely on dataclass __init__

        try:
            return cls(**processed_data)
        except TypeError as e:
            raise ValueError(f"Error creating {cls.__name__} from dict: {e}. Data: {processed_data}")


    def save_to_json(self, filepath: Union[str, Path]) -> None:
        """Save the configuration to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_from_json(cls: Type[T], filepath: Union[str, Path]) -> T:
        """Load the configuration from a JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
        instance = cls.from_dict(data)
        instance.validate() # Validate after loading
        return instance

    def __post_init__(self) -> None:
        """
        Called after __init__. Useful for complex initialization or validation
        that depends on multiple fields.
        """
        self.validate() # Call validate after object is initialized

    def update(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration fields from a dictionary.
        Performs validation after update.
        """
        for key, value in new_config.items():
            if hasattr(self, key):
                # If the attribute is a nested BaseConfig, update it recursively
                current_value = getattr(self, key)
                if isinstance(current_value, BaseConfig) and isinstance(value, dict):
                    current_value.update(value)
                else:
                    # Potentially convert types here if needed, e.g., str to Path
                    field_type = self.__annotations__.get(key)
                    if field_type:
                        # Handle Optional types
                        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                             possible_types = [arg for arg in field_type.__args__ if arg is not type(None)]
                             if possible_types:
                                 field_type = possible_types[0]

                        if field_type is Path and isinstance(value, str):
                            setattr(self, key, Path(value))
                        elif is_dataclass(field_type) and isinstance(value, dict):
                             # If it's a non-BaseConfig dataclass, try to reconstruct
                             if hasattr(field_type, 'from_dict'):
                                 setattr(self, key, field_type.from_dict(value))
                             else:
                                 setattr(self, key, field_type(**value))
                        else:
                            setattr(self, key, value)
                    else:
                        setattr(self, key, value)
            else:
                # Handle unknown keys if necessary (e.g., raise error or warning)
                print(f"Warning: Unknown configuration key '{key}' during update.")

        self.validate() # Re-validate after updates

    def merge(self: T, other_config: T) -> T:
        """
        Merge another configuration object into this one.
        Fields from `other_config` that are not None will override fields in this config.
        Returns a new configuration object.
        """
        current_dict = self.to_dict()
        other_dict = other_config.to_dict()

        merged_dict = current_dict.copy()

        for key, value in other_dict.items():
            if value is not None:
                if key in merged_dict and isinstance(merged_dict[key], dict) and isinstance(value, dict):
                    # Potentially recursive merge for nested dicts if they represent nested configs
                    # This part needs to be smarter if we want deep merging of nested BaseConfig
                    field_type = self.__annotations__.get(key)
                    if hasattr(field_type, '__origin__') and field_type.__origin__ is Union: # Optional[T]
                        field_type = [arg for arg in field_type.__args__ if arg is not type(None)][0]

                    if issubclass(field_type, BaseConfig):
                        # Get the original nested config object to call its merge method
                        original_nested_config = getattr(self, key)
                        # Create a new instance of the nested config from the 'value' dict
                        other_nested_config = field_type.from_dict(value)
                        merged_dict[key] = original_nested_config.merge(other_nested_config).to_dict()
                    else:
                         merged_dict[key] = value # Overwrite for non-BaseConfig dicts or simple types
                else:
                    merged_dict[key] = value

        return self.__class__.from_dict(merged_dict)

# Example Usage:
# @dataclass
# class NestedConfig(BaseConfig):
#     param_c: str = "default_c"
#     param_d: int = 10

#     def validate(self) -> None:
#         super().validate()
#         if self.param_d < 0:
#             raise ValueError("param_d must be non-negative in NestedConfig")

# @dataclass
# class MyConfig(BaseConfig):
#     param_a: str = "default_a"
#     param_b: Optional[Path] = None
#     nested: NestedConfig = field(default_factory=NestedConfig)

#     def validate(self) -> None:
#         super().validate() # Good practice to call super's validate
#         if self.param_a == "invalid":
#             raise ValueError("param_a cannot be 'invalid'")
#         if self.param_b and not self.param_b.is_dir():
#             # Example: if param_b is provided, it must be a directory
#             # This specific validation might be too strict or context-dependent
#             # print(f"Warning: Path {self.param_b} is not a directory (if it's meant to be one).")
#             pass

# if __name__ == "__main__":
#     # Creation and Validation
#     try:
#         config1 = MyConfig()
#         print(f"Config 1 (default): {config1.to_dict()}")
#         config1.validate() # validate is called in __post_init__

#         config2 = MyConfig(param_a="custom_a", param_b=Path("/tmp/test_path"), nested=NestedConfig(param_d=20))
#         print(f"Config 2 (custom): {config2.to_dict()}")

#         # Validation failure example
#         # config_invalid_a = MyConfig(param_a="invalid") # Raises ValueError

#         # nested_invalid = NestedConfig(param_d=-5) # Raises ValueError
#         # config_invalid_nested = MyConfig(nested=nested_invalid)

#     except ValueError as e:
#         print(f"Validation Error: {e}")

#     # Serialization / Deserialization
#     test_file = Path("test_config.json")
#     if config2:
#        config2.save_to_json(test_file)
#        loaded_config = MyConfig.load_from_json(test_file)
#        print(f"Loaded_config: {loaded_config.to_dict()}")
#        assert loaded_config == config2
#        test_file.unlink() # Clean up

#     # from_dict and to_dict
#     dict_data = {"param_a": "from_dict_a", "param_b": "/path/to/data", "nested": {"param_c": "nested_c", "param_d": 30}}
#     config_from_dict = MyConfig.from_dict(dict_data)
#     print(f"Config from dict: {config_from_dict.to_dict()}")
#     assert config_from_dict.param_b == Path("/path/to/data")
#     assert config_from_dict.nested.param_d == 30

#     # Update
#     config_to_update = MyConfig()
#     print(f"Before update: {config_to_update.to_dict()}")
#     config_to_update.update({"param_a": "updated_a", "nested": {"param_d": 100}, "param_b": "new/path"})
#     print(f"After update: {config_to_update.to_dict()}")
#     assert config_to_update.param_a == "updated_a"
#     assert config_to_update.nested.param_d == 100
#     assert config_to_update.param_b == Path("new/path")

#     # Merge
#     cfg_base = MyConfig(param_a="base_a", param_b=Path("base/path"), nested=NestedConfig(param_c="base_c", param_d=1))
#     cfg_override = MyConfig(param_a="override_a", param_b=None, nested=NestedConfig(param_c=None, param_d=2)) # param_b=None should not override, nested.param_c=None should not override

#     # Create a specific override dictionary for merge demonstration
#     override_dict_for_merge = {"param_a": "override_a", "nested": {"param_d": 2}}
#     # We need an actual MyConfig object for the merge method signature
#     cfg_for_merge_logic = MyConfig.from_dict(override_dict_for_merge)


#     merged_cfg = cfg_base.merge(cfg_for_merge_logic)
#     print(f"Base Cfg: {cfg_base.to_dict()}")
#     print(f"Override Cfg for merge: {cfg_for_merge_logic.to_dict()}")
#     print(f"Merged Cfg: {merged_cfg.to_dict()}")

#     assert merged_cfg.param_a == "override_a" # Overridden
#     assert merged_cfg.param_b == Path("base/path") # Not overridden by None
#     assert merged_cfg.nested.param_c == "base_c" # Not overridden by None in nested
#     assert merged_cfg.nested.param_d == 2 # Overridden in nested

#     print("BaseConfig tests passed.")
