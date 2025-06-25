from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Iterator, Union, Callable # Added Callable
from pathlib import Path

from torch.utils.data import Dataset # Using PyTorch Dataset as a base

from arc_env.data.loaders.base import BaseDataLoader, ARCTaskData
from arc_env.data.processors.base import BaseDataProcessor
from .arc_dataset import ARCDataset, ARCDatasetItem # Leverage the base ARCDataset
from arc_env.exceptions import ARCError, ConfigurationError


class ChallengeDatasetItem(ARCDatasetItem):
    """
    Represents a single item from a ChallengeDataset.
    It extends ARCDatasetItem (which is ARCTaskData) and can include
    additional metadata or structures specific to challenges.
    """
    challenge_name: Optional[str] = None
    difficulty_rating: Optional[float] = None # e.g., 0.0 to 1.0
    tags: List[str] = []
    # Other challenge-specific metadata could go here.

    # We need a way to construct this from ARCTaskData and extra metadata.
    # This is tricky because ARCTaskData is a Protocol.
    # Let's assume ARCDatasetItem is a concrete class or we make one here.
    # For now, let's define a concrete structure for ChallengeDatasetItem:

    def __init__(self, task_data: ARCTaskData, challenge_info: Optional[Dict[str, Any]] = None):
        # Copy fields from ARCTaskData protocol
        self.train = task_data.train # type: ignore
        self.test = task_data.test   # type: ignore
        self.task_id = task_data.task_id if hasattr(task_data, 'task_id') else None # type: ignore

        # Populate challenge-specific fields
        if challenge_info:
            self.challenge_name = challenge_info.get("challenge_name", self.task_id)
            self.difficulty_rating = challenge_info.get("difficulty_rating")
            self.tags = challenge_info.get("tags", [])
        else:
            self.challenge_name = self.task_id
            self.difficulty_rating = None
            self.tags = []


class ChallengeDataset(ARCDataset[ChallengeDatasetItem]): # type: ignore # mypy issue with generic Dataset
    """
    A dataset class specifically for a collection of ARC challenges.
    A "challenge" might be a curated subset of ARC tasks, possibly with
    additional metadata (difficulty, tags, descriptions).

    This class can extend ARCDataset by:
    - Loading challenge-specific metadata alongside task data.
    - Filtering tasks based on challenge criteria (e.g., by tag or difficulty).
    - Providing items as ChallengeDatasetItem, which includes this metadata.
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        challenge_metadata_loader: Optional[Callable[[str], Dict[str, Any]]] = None, # Loads metadata for a task_id
        task_processors: Optional[List[BaseDataProcessor[ARCTaskData, ARCTaskData]]] = None,
        cache_processed_tasks: bool = False,
        # Challenge-specific filtering options:
        filter_by_tags: Optional[List[str]] = None,
        min_difficulty: Optional[float] = None,
        max_difficulty: Optional[float] = None,
        # Or provide a list of specific task_ids that form this "challenge"
        include_task_ids: Optional[List[str]] = None,
    ):
        """
        Args:
            data_loader: Loads the raw ARC task data.
            challenge_metadata_loader: A callable that takes a task_id string and returns
                                       a dictionary of metadata for that challenge/task.
                                       If None, no extra metadata is loaded beyond task_id.
            task_processors: Processors for augmenting/validating task data.
            cache_processed_tasks: Whether to cache tasks after loading and processing.
            filter_by_tags: Only include tasks that have ALL of these tags.
            min_difficulty: Minimum difficulty rating to include.
            max_difficulty: Maximum difficulty rating to include.
            include_task_ids: Explicit list of task IDs to include in this dataset.
                              If provided, other filters might apply to this subset.
        """
        super().__init__(data_loader, task_processors, cache_processed_tasks) # Initializes self.task_ids

        self.metadata_loader = challenge_metadata_loader

        # Apply filters to self.task_ids if specified
        original_task_count = len(self.task_ids)

        if include_task_ids is not None:
            # Filter self.task_ids to only those in include_task_ids
            # This also preserves the order from include_task_ids if desirable.
            # For simplicity, just filter:
            valid_ids_set = set(include_task_ids)
            self.task_ids = [tid for tid in self.task_ids if tid in valid_ids_set]
            if not self.task_ids and include_task_ids: # If include_task_ids was non-empty but result is empty
                 print(f"Warning: None of the specified include_task_ids were found in the base loader's tasks.")


        # Further filtering based on metadata (tags, difficulty)
        # This requires loading metadata for each task before filtering.
        # This can be slow if metadata_loader is expensive and not cached.
        if filter_by_tags or min_difficulty is not None or max_difficulty is not None:
            if not self.metadata_loader:
                raise ConfigurationError(
                    "Filtering by tags or difficulty requires a challenge_metadata_loader to be provided."
                )

            filtered_ids = []
            for task_id in self.task_ids: # Iterate over potentially already subsetted task_ids
                metadata = self.metadata_loader(task_id) # type: ignore # Assume it's callable
                if not metadata: continue # Skip if no metadata

                # Tag filtering
                if filter_by_tags:
                    task_tags = set(metadata.get("tags", []))
                    if not set(filter_by_tags).issubset(task_tags):
                        continue # Does not meet tag criteria

                # Difficulty filtering
                task_difficulty = metadata.get("difficulty_rating")
                if task_difficulty is not None: # Only filter if difficulty is present
                    if min_difficulty is not None and task_difficulty < min_difficulty:
                        continue
                    if max_difficulty is not None and task_difficulty > max_difficulty:
                        continue
                elif min_difficulty is not None or max_difficulty is not None:
                    # If difficulty filter is active but task has no difficulty, exclude it.
                    continue

                filtered_ids.append(task_id)
            self.task_ids = filtered_ids

        if len(self.task_ids) < original_task_count:
            print(f"ChallengeDataset: Filtered tasks from {original_task_count} down to {len(self.task_ids)}.")
        if not self.task_ids:
            print("Warning: ChallengeDataset is empty after applying filters.")


    def __getitem__(self, idx: int) -> ChallengeDatasetItem:
        """
        Retrieves the task at the given index, processes it, and wraps it
        as a ChallengeDatasetItem with associated metadata.
        """
        # Get the processed ARCTaskData using the parent's __getitem__
        # This handles loading, base processing, and caching.
        processed_task_data: ARCTaskData = super().__getitem__(idx) # type: ignore # ARCDatasetItem is ARCTaskData

        task_id = self.task_ids[idx] # Get task_id for this index again for metadata loading

        challenge_info: Optional[Dict[str, Any]] = None
        if self.metadata_loader:
            try:
                challenge_info = self.metadata_loader(task_id)
            except Exception as e:
                print(f"Warning: Failed to load metadata for task '{task_id}': {e}")
                challenge_info = {"error": str(e)} # Include error in info if needed

        # Create the ChallengeDatasetItem
        # The constructor for ChallengeDatasetItem handles merging task_data and challenge_info
        challenge_item = ChallengeDatasetItem(processed_task_data, challenge_info)

        return challenge_item


# Example Usage:
# if __name__ == "__main__":
#     from arc_env.data.loaders.arc_loader import ARCFileLoader
#     import json # For dummy file creation
#     from pathlib import Path
#     import shutil # For cleanup

#     # 1. Create dummy task files
#     dummy_challenge_dir = Path("temp_challenge_dataset_tasks")
#     dummy_challenge_dir.mkdir(exist_ok=True)
#     tasks_content = {
#         "c_task_01": {"train": [{"input": [[1]], "output": [[0]]}], "test": [{"input": [[1]], "output": [[0]]}]},
#         "c_task_02": {"train": [{"input": [[2]], "output": [[3]]}], "test": [{"input": [[2]], "output": [[3]]}]},
#         "c_task_03": {"train": [{"input": [[4]], "output": [[5]]}], "test": [{"input": [[4]], "output": [[5]]}]},
#     }
#     for tid, content in tasks_content.items():
#         with open(dummy_challenge_dir / f"{tid}.json", "w") as f: json.dump(content, f)

#     # 2. Create a dummy metadata loader function
#     dummy_metadata = {
#         "c_task_01": {"challenge_name": "Challenge Alpha", "difficulty_rating": 0.2, "tags": ["symmetry", "color"]},
#         "c_task_02": {"challenge_name": "Challenge Beta", "difficulty_rating": 0.5, "tags": ["geometry", "counting"]},
#         "c_task_03": {"challenge_name": "Challenge Gamma", "difficulty_rating": 0.8, "tags": ["pattern", "color", "geometry"]},
#     }
#     def load_dummy_metadata(task_id: str) -> Dict[str, Any]:
#         return dummy_metadata.get(task_id, {})

#     try:
#         # 3. Initialize loader
#         loader = ARCFileLoader(data_directory=dummy_challenge_dir)

#         # 4. Create ChallengeDataset with filters
#         challenge_ds = ChallengeDataset(
#             data_loader=loader,
#             challenge_metadata_loader=load_dummy_metadata,
#             filter_by_tags=["geometry"], # Should pick c_task_02 and c_task_03
#             min_difficulty=0.4           # Should further filter to c_task_02 (0.5) and c_task_03 (0.8)
#         )
#         print(f"Number of tasks in ChallengeDataset (filtered by tag 'geometry', min_diff 0.4): {len(challenge_ds)}")
#         print(f"Task IDs in filtered dataset: {[tid for tid in challenge_ds.task_ids]}")


#         if len(challenge_ds) > 0:
#             print("\nIterating through ChallengeDataset:")
#             for item in challenge_ds:
#                 print(f"  Task ID: {item.task_id}, Challenge: {item.challenge_name}, Diff: {item.difficulty_rating}, Tags: {item.tags}")
#                 print(f"    Train input 0: {item.train[0]['input']}") # type: ignore

#         # Test with include_task_ids
#         challenge_ds_specific_ids = ChallengeDataset(
#             data_loader=loader,
#             challenge_metadata_loader=load_dummy_metadata,
#             include_task_ids=["c_task_01", "c_task_03"], # Only these two
#             filter_by_tags=["color"] # Both have "color" tag
#         )
#         print(f"\nChallengeDataset (specific IDs ['c_task_01', 'c_task_03'], tag 'color'): {len(challenge_ds_specific_ids)}")
#         print(f"Task IDs: {[tid for tid in challenge_ds_specific_ids.task_ids]}")
#         for item in challenge_ds_specific_ids:
#              print(f"  Task ID: {item.task_id}, Challenge: {item.challenge_name}")


#     except (ARCError, ConfigurationError) as e:
#         print(f"ChallengeDataset Error: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#     finally:
#         shutil.rmtree(dummy_challenge_dir, ignore_errors=True)
