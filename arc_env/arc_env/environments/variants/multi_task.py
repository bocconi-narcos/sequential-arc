from __future__ import annotations # For forward reference to ARCEnv if needed, and other hints

from typing import Optional, Dict, Any, List, Tuple # Tuple not directly used in signatures here
import random # For sample_task_on_reset if np_random is not used from env

from arc_env.environments.arc_env import ARCEnv
from arc_env.config.environment import EnvironmentConfig
from arc_env.config.action_space import ActionSpaceConfig
from arc_env.data.loaders.base import BaseDataLoader # For type hints
from arc_env.dsl.core.operation_registry import OperationRegistry # For type hints
from arc_env.exceptions import ConfigurationError, TaskNotSetError, ARCError # Added ARCError
from arc_env.spaces.observation_spaces import ARCStructuredObservation # For return type hint

# Note: The CurriculumWrapper provides a way to sequence tasks.
# This MultiTaskARCEnv could offer alternative multi-task behaviors:
# 1. Randomly sampling a task on each reset from a predefined set.
# 2. Modifying the observation space to include task identifiers or embeddings.
# 3. Handling rewards averaged over multiple tasks or a distribution of tasks.

class MultiTaskARCEnv(ARCEnv):
    """
    A variant of ARCEnv designed for multi-task scenarios.

    This environment can manage a collection of ARC tasks and switch between them,
    potentially based on a predefined sequence, random sampling, or curriculum.
    It might also modify observations to include task-specific information if needed
    for meta-learning or multi-task agents.

    If `task_id_sequence` is provided, it behaves like a curriculum, resetting to the
    next task in the sequence. If `sample_task_on_reset` is True, a random task
    from the available pool is chosen on each reset.
    """

    def __init__(
        self,
        env_config: Optional[EnvironmentConfig] = None,
        action_space_config: Optional[ActionSpaceConfig] = None,
        data_loader: Optional[BaseDataLoader] = None, # Must be able to list many tasks
        operation_registry: Optional[OperationRegistry] = None,

        # Multi-task specific configurations:
        task_id_pool: Optional[List[str]] = None, # Explicit list of task IDs to use
        task_id_sequence: Optional[List[str]] = None, # Fixed sequence of tasks
        sample_task_on_reset: bool = False, # If true, randomly pick from pool/available on reset
        loop_sequence: bool = True, # If using task_id_sequence, loop when it ends
        # initial_task_id is inherited from ARCEnv, but behavior might be overridden by sequence/sampling
    ):
        super().__init__(
            env_config=env_config,
            action_space_config=action_space_config,
            data_loader=data_loader,
            operation_registry=operation_registry,
            initial_task_id=None # MultiTaskARCEnv will handle initial task selection
        )

        if task_id_pool and task_id_sequence:
            print("Warning: Both task_id_pool and task_id_sequence provided. task_id_sequence will be used.")

        self.all_available_task_ids_from_loader: List[str] = self.data_loader.list_available_tasks()
        if not self.all_available_task_ids_from_loader:
            raise ConfigurationError("MultiTaskARCEnv: Data loader found no available tasks.")

        if task_id_sequence:
            self.active_task_ids = [tid for tid in task_id_sequence if tid in self.all_available_task_ids_from_loader]
            if len(self.active_task_ids) != len(task_id_sequence):
                print(f"Warning: Some tasks in task_id_sequence were not found in data_loader. Using {len(self.active_task_ids)} valid tasks for sequence.")
            if not self.active_task_ids: raise ConfigurationError("Task sequence resulted in no valid tasks.")
            self.use_sequence = True
            self.sample_task_on_reset = False # Sequence overrides random sampling
        elif task_id_pool:
            self.active_task_ids = [tid for tid in task_id_pool if tid in self.all_available_task_ids_from_loader]
            if len(self.active_task_ids) != len(task_id_pool):
                 print(f"Warning: Some tasks in task_id_pool were not found in data_loader. Using {len(self.active_task_ids)} valid tasks for pool.")
            if not self.active_task_ids: raise ConfigurationError("Task pool resulted in no valid tasks.")
            self.use_sequence = False
            self.sample_task_on_reset = sample_task_on_reset # Respect this flag if pool is given
        else: # No specific pool or sequence, use all tasks from loader
            self.active_task_ids = self.all_available_task_ids_from_loader
            self.use_sequence = False
            self.sample_task_on_reset = sample_task_on_reset

        if not self.active_task_ids:
            raise ConfigurationError("MultiTaskARCEnv: No active tasks to run (pool/sequence is empty or invalid).")

        self.current_sequence_idx: int = -1 # For sequence mode
        self.loop_sequence: bool = loop_sequence

        # Set the first task based on mode
        if self.use_sequence:
            self.current_sequence_idx = 0 # Will be used in first reset
            self.set_task(self.active_task_ids[self.current_sequence_idx])
        elif self.sample_task_on_reset:
            # Random task will be chosen in the first reset call.
            # Set a placeholder task for now, or the first from active_task_ids.
            self.set_task(self.active_task_ids[0])
        else: # Default: use first task in active_task_ids (could be from pool or all)
            self.set_task(self.active_task_ids[0])

        print(f"MultiTaskARCEnv initialized. Manages {len(self.active_task_ids)} tasks. "
              f"Mode: {'Sequence' if self.use_sequence else ('Random Sample' if self.sample_task_on_reset else 'Fixed First Task')}.")

        # Observation space modification (optional):
        # If we want to add task_id directly into the observation an agent sees:
        # This would require changing self.observation_space (e.g., adding a Discrete space for task_id).
        # For now, task_id is in the `info` dict.

    def _select_next_task_id(self) -> str:
        """Determines the next task_id based on the multi-task mode."""
        if self.use_sequence:
            self.current_sequence_idx += 1
            if self.current_sequence_idx >= len(self.active_task_ids):
                if self.loop_sequence:
                    self.current_sequence_idx = 0
                else:
                    # Curriculum finished. What to do?
                    # Option: keep serving the last task.
                    print("MultiTaskARCEnv: Task sequence finished (not looping). Using last task.")
                    self.current_sequence_idx = len(self.active_task_ids) - 1
                    # Could also raise an error or have a completion flag.
            return self.active_task_ids[self.current_sequence_idx]

        elif self.sample_task_on_reset:
            if not self.active_task_ids: # Should not happen if constructor checks pass
                raise ARCError("No tasks available for random sampling.")
            return self.np_random.choice(self.active_task_ids) # gym.Env provides self.np_random

        else: # Default behavior: stick to the current task unless changed by options.
              # This mode is less "multi-task" unless options in reset are used.
              # If an initial task was set, it will be that one.
            if self.current_task_id is None: # Should have been set in __init__
                 return self.active_task_ids[0]
            return self.current_task_id


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ARCStructuredObservation, Dict[str, Any]]:
        super(ARCEnv, self).reset(seed=seed) # Call grandparent's reset for seeding, skip ARCEnv's reset task logic.
                                             # This is a bit of a hack to control task setting.
                                             # A cleaner way is for ARCEnv.reset to check if options['task_id'] is already handled.

        new_task_id_selected_by_wrapper = False
        if options and "task_id" in options:
            # If task_id is explicitly provided in options, use it.
            requested_task_id = options["task_id"]
            if requested_task_id in self.all_available_task_ids_from_loader: # Check against all, not just active_task_ids
                if requested_task_id != self.current_task_id:
                    self.set_task(requested_task_id) # This also resets current_test_input_idx
                new_task_id_selected_by_wrapper = True # Task was set by option
                 # If this task was part of a sequence, update sequence_idx if possible
                if self.use_sequence and requested_task_id in self.active_task_ids:
                    try: self.current_sequence_idx = self.active_task_ids.index(requested_task_id)
                    except ValueError: pass # Not in current sequence, but valid task
            else:
                print(f"Warning: Task ID '{requested_task_id}' in reset options not found by loader. Ignoring.")

        if not new_task_id_selected_by_wrapper:
            # If options didn't specify a task, or if this env manages sequence/sampling:
            next_task_id = self._select_next_task_id()
            if next_task_id != self.current_task_id: # Avoid redundant set_task
                 self.set_task(next_task_id)
            # If next_task_id is same as current, set_task is skipped, current_test_input_idx remains.
            # If new task, current_test_input_idx is reset to 0 by set_task.

        # Handle test_input_idx from options if task wasn't changed by options['task_id']
        # or if it was changed but we still want to respect test_input_idx.
        if options and "test_input_idx" in options:
            new_idx = options["test_input_idx"]
            if not self.current_task_data: raise TaskNotSetError("Cannot set test_input_idx: no task loaded.")
            if 0 <= new_idx < self.num_test_inputs_in_task:
                self.current_test_input_idx = new_idx
            else:
                print(f"Warning: Invalid test_input_idx {new_idx} in options. Using current: {self.current_test_input_idx}")

        # Now that the task is set (either by options or by multi-task logic),
        # proceed with the rest of the reset logic from ARCEnv.
        if not self.current_task_id or not self.current_task_data: # Should be set by now
            raise TaskNotSetError("MultiTaskARCEnv.reset(): Task ID is not set after selection logic.")

        self._initialize_episode_state() # From ARCEnv, sets up current_grid etc.

        obs = self._get_observation()
        info = self._get_info() # From ARCEnv
        info["multi_task_mode"] = "sequence" if self.use_sequence else ("random_sample" if self.sample_task_on_reset else "fixed_or_option_driven")
        if self.use_sequence:
            info["sequence_idx"] = self.current_sequence_idx
            info["sequence_looping"] = self.loop_sequence

        return obs, info

    # Step, render, close, get_challenge_info are inherited from ARCEnv.
    # They will operate on the self.current_task_id set by this wrapper's reset logic.

# Example Usage:
# if __name__ == "__main__":
#     from pathlib import Path
#     import json
#     import shutil
#     from arc_env.data.loaders.arc_loader import ARCFileLoader

#     dummy_mt_task_dir = Path("temp_multi_task_arc_tasks")
#     dummy_mt_task_dir.mkdir(exist_ok=True)
#     task_ids_for_multi = ["mt_task_alpha", "mt_task_beta", "mt_task_gamma"]
#     for tid_m in task_ids_for_multi:
#         content = {"train": [{"input": [[hash(tid_m)%10]], "output": [[0]]}], "test": [{"input": [[1]], "output": [[0]]}]}
#         with open(dummy_mt_task_dir / f"{tid_m}.json", "w") as f: json.dump(content, f)

#     mt_loader = ARCFileLoader(data_directory=dummy_mt_task_dir)
#     mt_env_cfg = EnvironmentConfig(data_path=str(dummy_mt_task_dir)) # For superclass if it re-initializes loader

#     try:
#         print("--- Testing MultiTaskARCEnv (Sequence Mode) ---")
#         multi_env_seq = MultiTaskARCEnv(
#             env_config=mt_env_cfg, # Pass config so ARCEnv doesn't try to load default assets
#             data_loader=mt_loader,
#             task_id_sequence=task_ids_for_multi,
#             loop_sequence=True
#         )
#         for i in range(len(task_ids_for_multi) + 1): # Test looping
#             obs, info = multi_env_seq.reset()
#             print(f"Episode {i+1}: Task ID = {info.get('task_id')}, Seq Idx = {info.get('sequence_idx')}")
#             assert info.get('task_id') == task_ids_for_multi[info.get('sequence_idx')]

#         print("\n--- Testing MultiTaskARCEnv (Random Sample Mode) ---")
#         multi_env_rand = MultiTaskARCEnv(
#             env_config=mt_env_cfg,
#             data_loader=mt_loader,
#             task_id_pool=task_ids_for_multi, # Could also let it use all from loader
#             sample_task_on_reset=True
#         )
#         print(f"Task pool for random sampling: {multi_env_rand.active_task_ids}")
#         for i in range(5):
#             obs, info = multi_env_rand.reset()
#             print(f"Random Episode {i+1}: Task ID = {info.get('task_id')}")
#             assert info.get('task_id') in task_ids_for_multi

#         print("\n--- Testing MultiTaskARCEnv (Reset with Options) ---")
#         multi_env_opts = MultiTaskARCEnv(
#             env_config=mt_env_cfg,
#             data_loader=mt_loader,
#             task_id_pool=task_ids_for_multi # Initial pool
#         )
#         # Initial reset (will pick first from pool, e.g. mt_task_alpha)
#         _, info_init = multi_env_opts.reset()
#         print(f"Initial task: {info_init.get('task_id')}")

#         # Reset to a specific task via options
#         target_task = "mt_task_gamma"
#         if target_task in multi_env_opts.active_task_ids:
#             _, info_specific = multi_env_opts.reset(options={"task_id": target_task})
#             print(f"Reset to specific task via options: {info_specific.get('task_id')}")
#             assert info_specific.get('task_id') == target_task
#         else:
#             print(f"Skipping specific task test, '{target_task}' not in active pool.")


#     except (ConfigurationError, TaskNotSetError, ARCError) as e:
#         print(f"MultiTaskARCEnv Error: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         import traceback
#         traceback.print_exc()

#     finally:
#         if dummy_mt_task_dir.exists():
#             shutil.rmtree(dummy_mt_task_dir)
