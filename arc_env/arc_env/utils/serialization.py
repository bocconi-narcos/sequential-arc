import json
import pickle
from pathlib import Path
from typing import Any, Union, Optional, Dict, List
import numpy as np

# Utilities for serializing and deserializing common data structures,
# including those involving NumPy arrays.

class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy data types (ndarray, np.integer, np.floating, np.bool_).
    Converts them to their Python native equivalents for JSON serialization.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist() # Convert ndarrays to Python lists
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Path): # Handle Path objects if they appear
            return str(obj)
        return super().default(obj)

def save_to_json(
    data: Union[Dict[Any, Any], List[Any]],
    filepath: Union[str, Path],
    indent: Optional[int] = 4,
    use_np_encoder: bool = True,
    **kwargs: Any
) -> None:
    """
    Saves data (dict or list) to a JSON file.

    Args:
        data: The data to save.
        filepath: Path to the output JSON file.
        indent: Indentation level for pretty printing. Default is 4.
                Set to None for compact output.
        use_np_encoder: If True, use the custom NpEncoder to handle NumPy types.
        **kwargs: Additional arguments to pass to json.dump().
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    encoder_class = NpEncoder if use_np_encoder else None

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, cls=encoder_class, **kwargs)
    print(f"Data successfully saved to JSON: {filepath}")


def load_from_json(filepath: Union[str, Path], **kwargs: Any) -> Union[Dict[Any, Any], List[Any]]:
    """
    Loads data from a JSON file.

    Args:
        filepath: Path to the JSON file.
        **kwargs: Additional arguments to pass to json.load().

    Returns:
        The loaded data (typically a dict or list).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f, **kwargs)
    # print(f"Data successfully loaded from JSON: {filepath}")
    return data


def save_to_pickle(data: Any, filepath: Union[str, Path], **kwargs: Any) -> None:
    """
    Saves data to a pickle file.
    Warning: Pickle files can be insecure if loaded from untrusted sources.

    Args:
        data: The Python object to pickle.
        filepath: Path to the output pickle file.
        **kwargs: Additional arguments to pass to pickle.dump().
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f: # Open in binary mode for pickle
        pickle.dump(data, f, **kwargs)
    print(f"Data successfully saved to pickle: {filepath}")


def load_from_pickle(filepath: Union[str, Path], **kwargs: Any) -> Any:
    """
    Loads data from a pickle file.
    Warning: Only load pickle files from trusted sources due to security risks.

    Args:
        filepath: Path to the pickle file.
        **kwargs: Additional arguments to pass to pickle.load().

    Returns:
        The unpickled Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        pickle.UnpicklingError: If there's an error during unpickling.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")

    with open(filepath, 'rb') as f: # Open in binary mode
        data = pickle.load(f, **kwargs)
    # print(f"Data successfully loaded from pickle: {filepath}")
    return data


# Example Usage:
if __name__ == "__main__":
    temp_dir = Path("_temp_serialization_tests")
    temp_dir.mkdir(exist_ok=True)

    # --- JSON Serialization Example ---
    json_file_path = temp_dir / "test_data.json"
    data_with_numpy = {
        "name": "ARC Task Sample",
        "id": np.int64(12345), # NumPy integer
        "version": np.float32(1.2), # NumPy float
        "completed": np.bool_(True), # NumPy boolean
        "input_grid": np.array([[1, 0, 2], [0, 3, 0]]),
        "output_grid": np.array([[5, 5, 5]]),
        "metadata": {"source": Path("path/to/source_file.json")} # Path object
    }

    print("\n--- Testing JSON Serialization (with NpEncoder) ---")
    try:
        save_to_json(data_with_numpy, json_file_path)
        loaded_json_data = load_from_json(json_file_path)

        print("Original data (relevant parts):")
        print(f"  ID type: {type(data_with_numpy['id'])}, Value: {data_with_numpy['id']}")
        print(f"  Input grid type: {type(data_with_numpy['input_grid'])}, Shape: {data_with_numpy['input_grid'].shape}")
        print(f"  Metadata source type: {type(data_with_numpy['metadata']['source'])}")


        print("\nLoaded JSON data (relevant parts, types will be Python native):")
        print(f"  ID type: {type(loaded_json_data['id'])}, Value: {loaded_json_data['id']}") # Should be int
        # input_grid_loaded = np.array(loaded_json_data['input_grid']) # Convert list back to ndarray
        # print(f"  Input grid (re-NumPy'd) type: {type(input_grid_loaded)}, Shape: {input_grid_loaded.shape}")
        print(f"  Input grid (as list from JSON) type: {type(loaded_json_data['input_grid'])}")
        print(f"  Metadata source type: {type(loaded_json_data['metadata']['source'])}") # Should be str

        # Check if content matches (after converting list back to array for grid comparison)
        assert loaded_json_data['id'] == data_with_numpy['id']
        assert np.array_equal(np.array(loaded_json_data['input_grid']), data_with_numpy['input_grid'])
        assert str(data_with_numpy['metadata']['source']) == loaded_json_data['metadata']['source']
        print("JSON load/save content matches original (after type conversions).")

    except Exception as e:
        print(f"Error during JSON serialization test: {e}")
    finally:
        if json_file_path.exists(): json_file_path.unlink()


    # --- Pickle Serialization Example ---
    pickle_file_path = temp_dir / "test_data.pkl"
    # data_with_numpy can be used directly for pickle as it handles NumPy types.

    print("\n--- Testing Pickle Serialization ---")
    try:
        save_to_pickle(data_with_numpy, pickle_file_path)
        loaded_pickle_data = load_from_pickle(pickle_file_path)

        print("Original data (relevant parts):")
        print(f"  ID type: {type(data_with_numpy['id'])}, Value: {data_with_numpy['id']}")
        print(f"  Input grid type: {type(data_with_numpy['input_grid'])}, Shape: {data_with_numpy['input_grid'].shape}")
        print(f"  Metadata source type: {type(data_with_numpy['metadata']['source'])}")


        print("\nLoaded Pickle data (types should be preserved):")
        print(f"  ID type: {type(loaded_pickle_data['id'])}, Value: {loaded_pickle_data['id']}")
        print(f"  Input grid type: {type(loaded_pickle_data['input_grid'])}, Shape: {loaded_pickle_data['input_grid'].shape}")
        print(f"  Metadata source type: {type(loaded_pickle_data['metadata']['source'])}")


        assert type(loaded_pickle_data['id']) == type(data_with_numpy['id'])
        assert np.array_equal(loaded_pickle_data['input_grid'], data_with_numpy['input_grid'])
        assert type(loaded_pickle_data['input_grid']) == np.ndarray
        assert loaded_pickle_data['metadata']['source'] == data_with_numpy['metadata']['source'] # Path object equality
        print("Pickle load/save content and types match original.")

    except Exception as e:
        print(f"Error during Pickle serialization test: {e}")
    finally:
        if pickle_file_path.exists(): pickle_file_path.unlink()
        if temp_dir.exists(): temp_dir.rmdir() # Clean up directory if empty

    print("\nSerialization utilities tests finished.")
