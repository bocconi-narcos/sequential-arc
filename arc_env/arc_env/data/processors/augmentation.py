import numpy as np
from typing import Any, List, Dict, Callable, Tuple, Optional
import random

from .base import BaseDataProcessor
from arc_env.data.loaders.base import ARCTaskData # Protocol
from arc_env.dsl.utils import grid_utils # For potential advanced augmentations
from arc_env.exceptions import DataProcessingError

# Type for an augmentation function: takes a grid, returns an augmented grid
GridAugmentationFn = Callable[[np.ndarray], np.ndarray]

class GridAugmenter(BaseDataProcessor[np.ndarray, np.ndarray]):
    """
    A data processor that applies a sequence of augmentations to a single grid.
    """
    def __init__(self, augmentations: Optional[List[GridAugmentationFn]] = None, **kwargs: Any):
        """
        Args:
            augmentations: A list of augmentation functions to apply.
                           If None, no augmentations are applied by default.
        """
        super().__init__(**kwargs)
        self.augmentations = augmentations if augmentations is not None else []

    def process(self, data: np.ndarray) -> np.ndarray:
        """Applies the configured augmentations to the grid."""
        augmented_grid = np.copy(data) # Start with a copy
        for aug_fn in self.augmentations:
            try:
                augmented_grid = aug_fn(augmented_grid)
            except Exception as e:
                # Depending on severity, either raise, log, or skip this augmentation
                print(f"Warning: Augmentation function {aug_fn.__name__ if hasattr(aug_fn, '__name__') else aug_fn} failed: {e}")
                # To be robust, could return the grid state before failure or original grid.
                # For now, continue with the grid as it was before this failing augmentation.
        return augmented_grid

class ARCTaskAugmenter(BaseDataProcessor[ARCTaskData, ARCTaskData]):
    """
    A data processor that augments ARC task data.
    It can apply specified augmentations to the input and/or output grids
    of the training and test pairs.
    """
    # Define concrete task data type for return, matching ARCTaskData protocol
    from dataclasses import dataclass, field
    @dataclass
    class _AugmentedTaskData:
        train: List[Dict[str, np.ndarray]] = field(default_factory=list)
        test: List[Dict[str, np.ndarray]] = field(default_factory=list)
        task_id: Optional[str] = None

    def __init__(
        self,
        input_augmentations: Optional[List[GridAugmentationFn]] = None,
        output_augmentations: Optional[List[GridAugmentationFn]] = None,
        apply_to_train: bool = True,
        apply_to_test_inputs: bool = False, # Usually test inputs are fixed
        apply_to_test_outputs: bool = False, # Usually test outputs are fixed
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.input_augmenter = GridAugmenter(input_augmentations)
        self.output_augmenter = GridAugmenter(output_augmentations) # Can be same funcs or different
        self.apply_to_train = apply_to_train
        self.apply_to_test_inputs = apply_to_test_inputs
        self.apply_to_test_outputs = apply_to_test_outputs

    def process(self, data: ARCTaskData) -> ARCTaskData:
        """
        Applies augmentations to the grids within the ARCTaskData.
        Returns a new ARCTaskData object with augmented grids.
        """
        new_train_pairs = []
        if self.apply_to_train:
            for pair in data.train: # type: ignore
                new_input = self.input_augmenter.process(pair["input"])
                # Output augmentation should usually be consistent with input augmentation
                # if the goal is to preserve the task's transformation rule.
                # E.g., if input is rotated, output should also be rotated.
                # This simple setup applies them independently. More complex logic needed for linked augs.
                new_output = self.output_augmenter.process(pair["output"])
                new_train_pairs.append({"input": new_input, "output": new_output})
        else:
            new_train_pairs = [pair.copy() for pair in data.train] # type: ignore

        new_test_pairs = []
        for i, pair in enumerate(data.test): # type: ignore
            new_input = self.input_augmenter.process(pair["input"]) if self.apply_to_test_inputs else np.copy(pair["input"])
            new_output = self.output_augmenter.process(pair["output"]) if self.apply_to_test_outputs else np.copy(pair["output"])
            new_test_pairs.append({"input": new_input, "output": new_output})

        task_id = data.task_id if hasattr(data, 'task_id') else None
        return ARCTaskAugmenter._AugmentedTaskData(train=new_train_pairs, test=new_test_pairs, task_id=task_id) # type: ignore


# --- Example Grid Augmentation Functions ---
def random_flip(grid: np.ndarray, p_horizontal: float = 0.5, p_vertical: float = 0.5) -> np.ndarray:
    """Randomly flips the grid horizontally and/or vertically."""
    new_grid = grid
    if random.random() < p_horizontal:
        new_grid = np.fliplr(new_grid)
    if random.random() < p_vertical:
        new_grid = np.flipud(new_grid)
    return new_grid

def random_rotate90(grid: np.ndarray, p: float = 0.75) -> np.ndarray:
    """Randomly rotates the grid by 0, 90, 180, or 270 degrees if p is met."""
    if random.random() < p:
        k = random.randint(0, 3) # 0: no rotation, 1: 90deg ccw, 2: 180, 3: 270
        if k > 0 : # np.rot90(grid, 0) is grid.
             # Check if grid is square. Rotation of non-square can be problematic for fixed canvas.
            if grid.shape[0] != grid.shape[1]:
                # print(f"Warning: random_rotate90 applied to non-square grid {grid.shape}. Shape will change.")
                # Or, choose not to rotate non-square grids:
                return grid
            return np.rot90(grid, k)
    return grid

# More complex augmentations could involve:
# - Color permutations (e.g., swap red and blue, if consistent across task)
# - Noise injection (e.g., randomly change a few pixels)
# - Elastic deformations (less common for discrete ARC grids)
# - Resizing/padding variations (if canvas size can vary)


# Example Usage:
# if __name__ == "__main__":
#     from dataclasses import dataclass

#     @dataclass
#     class DummyTask: # Implements ARCTaskData protocol
#         train: List[Dict[str, np.ndarray]]
#         test: List[Dict[str, np.ndarray]]
#         task_id: Optional[str] = None

#     # Create a sample task
#     sample_grid_in = np.array([[1,2],[3,4]])
#     sample_grid_out = np.array([[4,3],[2,1]])
#     task = DummyTask(
#         task_id="aug_test_01",
#         train=[{"input": sample_grid_in, "output": sample_grid_out}],
#         test=[{"input": np.copy(sample_grid_in), "output": np.copy(sample_grid_out)}]
#     )

#     print("Original Task Data (Input):")
#     print(task.train[0]["input"])

#     # 1. GridAugmenter example
#     grid_aug = GridAugmenter(augmentations=[
#         lambda g: random_flip(g, p_horizontal=1.0, p_vertical=0.0), # Force horizontal flip
#         lambda g: random_rotate90(g, p=1.0) # Force some rotation
#     ])
#     augmented_single_grid = grid_aug.process(sample_grid_in)
#     print("\nSingle grid augmented (h-flipped + rotated):\n", augmented_single_grid)


#     # 2. ARCTaskAugmenter example
#     # Augment only inputs of training pairs with flips and rotations
#     task_augmenter = ARCTaskAugmenter(
#         input_augmentations=[random_flip, random_rotate90],
#         output_augmentations=None, # No augmentation on outputs for this example
#         apply_to_train=True,
#         apply_to_test_inputs=False
#     )

#     augmented_task = task_augmenter.process(task)
#     print("\nAugmented Task Data (Train Input possibly changed, Train Output same):")
#     print("Train Input (augmented):\n", augmented_task.train[0]["input"]) # type: ignore
#     print("Train Output (original):\n", augmented_task.train[0]["output"]) # type: ignore
#     print("Test Input (original):\n", augmented_task.test[0]["input"]) # type: ignore

#     # Note: For ARC, augmentations must often be applied consistently to
#     # input/output pairs and across examples to maintain task integrity.
#     # E.g., if you rotate one training input, its corresponding output must also be rotated.
#     # This requires a more sophisticated augmentation pipeline that can link augmentations.
#     # The current ARCTaskAugmenter applies them independently to input/output lists.
#     # A common strategy is to decide on an augmentation (e.g., "rotate by 90 deg")
#     # and apply that *same* transformation to all relevant grids in a task instance.
#     # This would require the process method to generate one random transform and apply it.
#     #
#     # Example of linked augmentation (conceptual):
#     # class ConsistentRotationAugmenter(ARCTaskAugmenter):
#     #     def process(self, data: ARCTaskData) -> ARCTaskData:
#     #         k = random.randint(0,3) # Choose one rotation for the whole task
#     #         def rotate_k(grid): return np.rot90(grid,k) if grid.shape[0]==grid.shape[1] else grid
#     #         self.input_augmenter.augmentations = [rotate_k]
#     #         self.output_augmenter.augmentations = [rotate_k]
#     #         return super().process(data)
#     #
#     # linked_aug = ConsistentRotationAugmenter() #Defaults to apply_to_train=True
#     # consistently_rotated_task = linked_aug.process(task)
#     # print("\nConsistently Rotated Task (Train Input & Output both rotated by same random k):")
#     # print("Train Input:\n", consistently_rotated_task.train[0]["input"])
#     # print("Train Output:\n", consistently_rotated_task.train[0]["output"])
