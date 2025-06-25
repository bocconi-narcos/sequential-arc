"""
Custom exceptions for the ARC Environment package.
"""

class ARCError(Exception):
    """Base class for all custom exceptions in the arc_env package."""
    pass

# --- Configuration System Exceptions ---
class ConfigurationError(ARCError):
    """Raised for errors related to configuration loading, validation, or access."""
    pass

# --- Data Management Exceptions ---
class DataLoadError(ARCError):
    """Raised when data loading fails (e.g., file not found, parsing error, validation error)."""
    pass

class DataProcessingError(ARCError):
    """Raised for errors occurring during data processing stages (e.g., augmentation, feature extraction)."""
    pass

# --- DSL System Exceptions ---
class OperationError(ARCError):
    """Raised for errors related to DSL operations (e.g., invalid parameters, unregistered operation)."""
    pass

class DSLExecutionError(ARCError):
    """Raised for errors occurring during the execution or interpretation of a DSL program/sequence."""
    pass

# --- Environment Exceptions ---
class TaskNotSetError(ARCError):
    """Raised when an environment operation requires a task to be set, but it hasn't been."""
    pass

class InvalidActionError(ARCError, ValueError): # Inherit from ValueError for compatibility with gym checks
    """Raised when an invalid action is attempted in the environment."""
    # gym.utils.play often expects ValueError for invalid actions.
    pass

# --- Solver System Exceptions ---
class SolverError(ARCError):
    """Base class for errors related to solvers."""
    pass

class ModelNotFoundError(SolverError, FileNotFoundError):
    """Raised when a solver tries to load a model file that doesn't exist."""
    pass

class SolverPredictionError(SolverError):
    """Raised when a solver fails to produce a prediction."""
    pass


# Example of how these might be used:
#
# def my_function_that_loads_data(path):
#     if not path.exists():
#         raise DataLoadError(f"Data file not found at {path}")
#     # ... load data ...
#
# def configure_component(config_dict):
#     if "required_key" not in config_dict:
#         raise ConfigurationError("Missing 'required_key' in configuration.")
#     # ... configure ...
#
# try:
#     configure_component({})
# except ConfigurationError as e:
#     print(f"Caught configuration error: {e}")

if __name__ == '__main__':
    print("ARC Custom Exceptions:")
    print(f"  ARCError: {ARCError('Base ARC error')}")
    print(f"  ConfigurationError: {ConfigurationError('Config issue')}")
    print(f"  DataLoadError: {DataLoadError('Failed to load')}")
    print(f"  DataProcessingError: {DataProcessingError('Failed to process')}")
    print(f"  OperationError: {OperationError('DSL op issue')}")
    print(f"  DSLExecutionError: {DSLExecutionError('DSL exec issue')}")
    print(f"  TaskNotSetError: {TaskNotSetError('Task not set')}")
    print(f"  InvalidActionError: {InvalidActionError('Bad action')}")
    print(f"  SolverError: {SolverError('Base solver error')}")
    print(f"  ModelNotFoundError: {ModelNotFoundError('Model file missing')}")
    print(f"  SolverPredictionError: {SolverPredictionError('Solver cannot predict')}")

    # Test inheritance
    try:
        raise InvalidActionError("Test action error")
    except ValueError as ve: # Catches InvalidActionError due to inheritance
        print(f"\nCaught InvalidActionError as ValueError: {ve}")
    except ARCError as ae: # Fallback if not caught as ValueError (should not happen here)
        print(f"Caught InvalidActionError as ARCError: {ae}")

    try:
        raise ModelNotFoundError("my_model.zip not found")
    except FileNotFoundError as fnf: # Catches ModelNotFoundError
        print(f"Caught ModelNotFoundError as FileNotFoundError: {fnf}")
    except SolverError as se:
        print(f"Caught ModelNotFoundError as SolverError: {se}")
