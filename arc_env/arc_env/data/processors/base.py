from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic

# Define generic types for input and output data of the processor
InputDataType = TypeVar('InputDataType')
OutputDataType = TypeVar('OutputDataType')

class BaseDataProcessor(ABC, Generic[InputDataType, OutputDataType]):
    """
    Abstract base class for data processors.

    Data processors are components that transform or analyze data.
    This could include:
    - Validation of data integrity or format.
    - Augmentation of data (e.g., rotating, flipping images for training).
    - Feature extraction.
    - Conversion between data formats.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the data processor with configuration options.
        """
        self.config = kwargs
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validates the configuration parameters passed during initialization.
        Subclasses can override this to define their expected parameters.
        """
        # Example:
        # if "expected_param" not in self.config:
        #     raise ValueError("Missing 'expected_param' in processor configuration.")
        pass # Default: no specific config validation

    @abstractmethod
    def process(self, data: InputDataType) -> OutputDataType:
        """
        Processes the input data and returns the transformed or analyzed output.

        Args:
            data: The input data to be processed.

        Returns:
            The processed data.

        Raises:
            DataProcessingError: If an error occurs during processing.
                                 (Assuming DataProcessingError in exceptions.py)
            NotImplementedError: If the method is not implemented.
        """
        pass

    def __call__(self, data: InputDataType) -> OutputDataType:
        """Makes the processor instance callable, delegating to the process method."""
        return self.process(data)

# Example (for illustration, not part of this file content):
# from arc_env.data.loaders.base import ARCTaskData # Assuming this path
# from arc_env.exceptions import DataProcessingError

# class MyTaskValidator(BaseDataProcessor[ARCTaskData, bool]):
#     """A processor that validates if task data is usable."""
#     def _validate_config(self) -> None:
#         # No specific config for this simple validator
#         pass

#     def process(self, data: ARCTaskData) -> bool:
#         """Checks if the task data has at least one train and one test pair."""
#         if not data.train or not data.test:
#             # Example of raising a specific processing error
#             # raise DataProcessingError("Task data must have at least one train and one test pair.")
#             print(f"Validation failed for task {data.task_id}: Missing train or test pairs.")
#             return False
#         # Add more checks as needed (e.g., grid sizes, color ranges)
#         print(f"Task {data.task_id} passed basic structural validation.")
#         return True

# if __name__ == '__main__':
#     # Dummy ARCTaskData for testing
#     @dataclasses.dataclass # Requires import
#     class DummyTask:
#         train: list
#         test: list
#         task_id: str = "dummy"

#     valid_task_data = DummyTask(train=[{"input": np.array([[1]]), "output": np.array([[1]])}],
#                                 test=[{"input": np.array([[0]]), "output": np.array([[0]])}])
#     invalid_task_data = DummyTask(train=[], test=[]) # No train/test pairs

#     validator = MyTaskValidator()
#     print(f"Processing valid task: {validator(valid_task_data)}")    # True
#     print(f"Processing invalid task: {validator(invalid_task_data)}") # False
#     # If DataProcessingError was raised instead of returning False:
#     # try:
#     #     validator(invalid_task_data)
#     # except DataProcessingError as e:
#     #     print(f"Caught expected error: {e}")
