from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Iterator, Union, cast # Added cast
from pathlib import Path

from torch.utils.data import Dataset # Common for PyTorch, but can be adapted for others

from arc_env.data.loaders.base import BaseDataLoader, ARCTaskData
from arc_env.data.loaders.arc_loader import ARCFileLoader # Default loader example
from arc_env.data.processors.base import BaseDataProcessor
from arc_env.exceptions import ARCError

# Define what a single item from this dataset might look like.
# This could be a full task, or a single input/output pair, or something else.
# For ARC, often a "task" is the unit.
class ARCDatasetItem(ARCTaskData): # Inherit from protocol for structure
    # Could add more fields specific to dataset interaction if needed
    pass


class ARCDataset(Dataset[ARCDatasetItem]):
    """
    A dataset class for the Abstraction and Reasoning Corpus (ARC).

    This class wraps a data loader and optional processors to provide
    an iterable interface over ARC tasks, compatible with frameworks
    like PyTorch's DataLoader.
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        # List of processors to apply sequentially to each loaded task
        task_processors: Optional[List[BaseDataProcessor[ARCTaskData, ARCTaskData]]] = None,
        # Alternatively, could have separate input_processors, output_processors for grids
        # Or a single processor that takes ARCTaskData and returns processed ARCTaskData.
        cache_processed_tasks: bool = False
    ):
        """
        Args:
            data_loader: An instance of a BaseDataLoader subclass to load raw task data.
            task_processors: An optional list of processors to apply to each task
                             after loading. Processors should take ARCTaskData and return
                             ARCTaskData (e.g., an ARCTaskAugmenter).
            cache_processed_tasks: If True, processed tasks are cached in memory to avoid
                                   reprocessing on subsequent accesses. Useful if processing
                                   is expensive (e.g. heavy augmentation).
        """
        if not isinstance(data_loader, BaseDataLoader): # Check against Protocol
             raise TypeError(f"data_loader must be an instance of BaseDataLoader, got {type(data_loader)}.")
        self.loader = data_loader

        self.task_processors = task_processors if task_processors is not None else []
        for i, proc in enumerate(self.task_processors):
            if not isinstance(proc, BaseDataProcessor):
                raise TypeError(f"Processor at index {i} is not a BaseDataProcessor.")

        self.cache_processed_tasks = cache_processed_tasks
        self._processed_tasks_cache: Dict[str, ARCDatasetItem] = {}

        try:
            self.task_ids: List[str] = self.loader.list_available_tasks() # Assumes task_ids are strings
            if not self.task_ids:
                print("Warning: ARCDataset initialized, but data loader found no available tasks.")
        except Exception as e:
            raise ARCError(f"Failed to list available tasks from data loader: {e}")


    def __len__(self) -> int:
        """Returns the total number of tasks in the dataset."""
        return len(self.task_ids)

    def __getitem__(self, idx: int) -> ARCDatasetItem:
        """
        Retrieves the task at the given index.
        The task is loaded and then processed by any configured task processors.
        """
        if not (0 <= idx < len(self.task_ids)):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.task_ids)} tasks.")

        task_id = self.task_ids[idx]

        if self.cache_processed_tasks and task_id in self._processed_tasks_cache:
            return self._processed_tasks_cache[task_id]

        try:
            # Load raw task data
            raw_task_data: ARCTaskData = self.loader.load_task(task_id)
        except Exception as e:
            raise ARCError(f"Failed to load task '{task_id}' at index {idx}: {e}")

        # Apply processors
        processed_task_data: ARCTaskData = raw_task_data
        if self.task_processors:
            for processor in self.task_processors:
                try:
                    # Assuming processors take ARCTaskData and return ARCTaskData
                    processed_task_data = processor.process(processed_task_data)
                except Exception as e:
                    raise ARCError(f"Processor {processor.__class__.__name__} failed for task '{task_id}': {e}")

        # Ensure it conforms to ARCDatasetItem (which is ARCTaskData)
        # This is more of a type system check; runtime conversion might be needed if processors change structure.
        # For now, assume processors maintain ARCTaskData compatibility.
        dataset_item = cast(ARCDatasetItem, processed_task_data)

        if self.cache_processed_tasks:
            self._processed_tasks_cache[task_id] = dataset_item

        return dataset_item

    def get_task_by_id(self, task_id: str) -> Optional[ARCDatasetItem]:
        """Retrieves a task by its string ID."""
        try:
            idx = self.task_ids.index(task_id)
            return self[idx]
        except ValueError: # task_id not in self.task_ids
            return None

    def __iter__(self) -> Iterator[ARCDatasetItem]:
        """Provides an iterator over the tasks in the dataset."""
        for i in range(len(self)):
            yield self[i]

# Example Usage:
# if __name__ == "__main__":
#     from arc_env.data.loaders.arc_loader import ARCFileLoader
#     from arc_env.data.processors.augmentation import ARCTaskAugmenter, random_flip
#     from dataclasses import dataclass # For ARCTaskData concrete type if needed by loader/processor directly

#     # Create dummy task files for ARCFileLoader
#     dummy_task_dir = Path("temp_arc_dataset_tasks")
#     dummy_task_dir.mkdir(exist_ok=True)
#     task_a_content = {"task_id":"task_a", "train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3]], "output": [[4]]}]}
#     task_b_content = {"task_id":"task_b", "train": [{"input": [[5]], "output": [[6]]}], "test": [{"input": [[7]], "output": [[8]]}]}
#     import json
#     with open(dummy_task_dir / "task_a.json", "w") as f: json.dump(task_a_content, f)
#     with open(dummy_task_dir / "task_b.json", "w") as f: json.dump(task_b_content, f)

#     try:
#         # 1. Initialize a data loader
#         loader = ARCFileLoader(data_directory=dummy_task_dir)

#         # 2. Initialize optional processors (e.g., an augmenter)
#         augmenter = ARCTaskAugmenter(
#             input_augmentations=[lambda g: random_flip(g, p_horizontal=1.0)], # Always flip inputs horizontally
#             apply_to_train=True
#         )
#         processors = [augmenter]

#         # 3. Create the dataset
#         arc_dataset = ARCDataset(data_loader=loader, task_processors=processors, cache_processed_tasks=True)

#         print(f"Number of tasks in dataset: {len(arc_dataset)}")

#         # Iterate through the dataset
#         print("\nIterating through dataset:")
#         for i, task_data_item in enumerate(arc_dataset):
#             print(f"Task {i} (ID: {task_data_item.task_id}):") # type: ignore
#             print(f"  Train input 0 (original was [[1]] or [[5]]): \n{task_data_item.train[0]['input']}") # type: ignore
#             print(f"  Train output 0 (should be original): \n{task_data_item.train[0]['output']}") # type: ignore

#         # Access by index
#         if len(arc_dataset) > 0:
#             print("\nAccessing first task by index:")
#             first_task = arc_dataset[0]
#             print(f"First task ID: {first_task.task_id}") # type: ignore

#         # Access by task ID
#         print("\nAccessing 'task_b' by ID:")
#         task_b = arc_dataset.get_task_by_id("task_b")
#         if task_b:
#             print(f"Task B train input (should be horizontally flipped [[5]] -> [[5]] if 1x1, or e.g. [[5,6]] -> [[6,5]]):")
#             print(task_b.train[0]['input']) # type: ignore
#         else:
#             print("Task 'task_b' not found by ID.")

#         # Test caching (access task_b again)
#         print("\nAccessing 'task_b' again (should be cached):")
#         task_b_cached = arc_dataset.get_task_by_id("task_b")
#         # If augmentation is random, caching ensures it's the same object / same augmented version.
#         if task_b_cached and task_b: # Check if both were found
#             # For random augmentations, this check might fail if not cached and re-augmented differently.
#             # With forced flip, it should be identical.
#             assert np.array_equal(task_b.train[0]['input'], task_b_cached.train[0]['input']), "Cached task differs!" # type: ignore
#             print("Cached task_b matches first retrieval.")


#     except ARCError as e:
#         print(f"ARCDataset Error: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#     finally:
#         import shutil
#         shutil.rmtree(dummy_task_dir, ignore_errors=True)
