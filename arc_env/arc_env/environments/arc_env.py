from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, Union, List, cast
import gymnasium as gym
import numpy as np
# import copy # Not strictly used now if np.copy is preferred for grids

from arc_env.core.base_env import BaseARCEnvironment
from arc_env.config.environment import EnvironmentConfig
from arc_env.config.action_space import ActionSpaceConfig
from arc_env.spaces.action_spaces import ARCActionSpace, DecodedArcOps
from arc_env.spaces.observation_spaces import ARCObservationSpace, ARCStructuredObservation
from arc_env.data.loaders.base import BaseDataLoader, ARCTaskData
from arc_env.dsl.core.operation_registry import OperationRegistry
from arc_env.dsl.core.base_operations import BaseOperation
from arc_env.dsl.utils.color_utils import ARC_BLACK # Default color for padding/canvas
from arc_env.exceptions import ARCError, ConfigurationError, TaskNotSetError

class ARCEnv(BaseARCEnvironment):
    """
    Main Gymnasium Environment for the Abstraction and Reasoning Corpus (ARC).

    This environment allows an agent to interact with ARC tasks by applying
    a sequence of operations to an input grid to transform it into an output grid.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"], # TODO: Implement these render modes
        "render_fps": 4, # Can be configured via EnvironmentConfig
    }

    def __init__(
        self,
        env_config: Optional[EnvironmentConfig] = None,
        action_space_config: Optional[ActionSpaceConfig] = None, # Config for ARCActionSpace
        data_loader: Optional[BaseDataLoader] = None,
        operation_registry: Optional[OperationRegistry] = None, # For ARCActionSpace
        initial_task_id: Optional[str] = None # Optionally load a task upon init
    ):
        super().__init__() # Calls BaseARCEnvironment init

        self.env_config = env_config if env_config else EnvironmentConfig()
        self.action_space_config = action_space_config if action_space_config else ActionSpaceConfig()

        if data_loader is None:
            # Provide a default loader if none is given (e.g., pointing to packaged assets)
            # This requires knowing where the assets are relative to this file.
            # For now, raise error if not provided, or make it configurable via env_config.data_path
            if self.env_config.data_path:
                from arc_env.data.loaders.arc_loader import ARCFileLoader # Default to individual files
                try:
                    self.data_loader = ARCFileLoader(self.env_config.data_path)
                except Exception as e:
                    raise ConfigurationError(
                        f"Failed to initialize default ARCFileLoader with data_path '{self.env_config.data_path}': {e}"
                    )
            else:
                raise ConfigurationError("ARCEnv requires a data_loader or a valid env_config.data_path.")
        else:
            if not isinstance(data_loader, BaseDataLoader): # Check protocol/ABC
                raise TypeError("Provided data_loader is not a valid BaseDataLoader.")
            self.data_loader = data_loader

        self.op_registry = operation_registry if operation_registry else OperationRegistry()
        # TODO: Populate op_registry with default operations and presets if it's empty.
        # This might involve DefaultPresetLoader, MinimalPresetLoader from dsl.presets.
        # For now, assume it's pre-populated if provided, or ARCActionSpace handles it.
        if not self.op_registry.list_available_presets() and not self.op_registry.list_available_operations():
             print("Warning: OperationRegistry is empty. ARCActionSpace may fail or be limited.")
             # Example: Load default preset if registry is empty
             try:
                 from arc_env.dsl.presets.default import DefaultPresetLoader
                 DefaultPresetLoader(self.op_registry).load()
                 print("Loaded DefaultPreset into OperationRegistry.")
             except ImportError:
                 print("Could not load DefaultPresetLoader. Ensure DSL presets are available.")


        # Initialize action and observation spaces
        self.action_space: ARCActionSpace = ARCActionSpace(self.action_space_config, self.op_registry)
        self.observation_space: ARCObservationSpace = ARCObservationSpace(self.env_config) # Uses env_config for canvas size

        self.render_mode = self.env_config.render_mode # From config
        if self.env_config.render_fps: ARCEnv.metadata["render_fps"] = self.env_config.render_fps


        # Environment state variables
        self.current_task_data: Optional[ARCTaskData] = None
        self.current_task_id: Optional[str] = None
        self.current_grid: np.ndarray = np.zeros( # Initial empty grid
            (self.env_config.canvas_size, self.env_config.canvas_size), dtype=np.uint8
        )
        self.current_selection_mask: Optional[np.ndarray] = None # For DSL ops that use selections
        self.target_grid: Optional[np.ndarray] = None # The solution grid for the current test input
        self.num_steps_taken: int = 0
        self.last_action_decoded: Optional[DecodedArcOps] = None
        self.last_reward: float = 0.0

        # For managing multiple test inputs within a single task
        self.current_test_input_idx: int = 0
        self.num_test_inputs_in_task: int = 0

        if initial_task_id:
            self.set_task(initial_task_id)
            # Note: set_task loads data but doesn't reset the env state like current_grid.
            # A reset is needed to fully initialize for a new task.
            # Consider if set_task should also trigger a self.reset() or partial reset.
            # For now, user must call reset() after __init__ or set_task if they want to start interaction.


    def set_task(self, task_id: str) -> None:
        """
        Loads a specific ARC task into the environment.
        This prepares the environment for a new problem, but `reset()` must be
        called to get the initial observation for that task.
        """
        try:
            task_data = self.data_loader.load_task(task_id)
            self.current_task_data = task_data
            self.current_task_id = task_id # Can also use task_data.task_id
            self.current_test_input_idx = 0 # Reset to the first test input of the new task

            if not self.current_task_data.test:
                raise TaskNotSetError(f"Task '{task_id}' loaded but has no test cases.")
            self.num_test_inputs_in_task = len(self.current_task_data.test)

            print(f"ARCEnv: Task '{task_id}' loaded. It has {len(self.current_task_data.train)} train pairs "
                  f"and {self.num_test_inputs_in_task} test inputs.")
            # Does not reset self.current_grid or self.num_steps_taken. That's for reset().
        except Exception as e: # Catch DataLoadError or other issues
            raise TaskNotSetError(f"Failed to set task '{task_id}': {e}")


    def _get_current_task_test_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the input and target output grid for the current test input
        index within the loaded task.
        """
        if not self.current_task_data or not self.current_task_data.test:
            raise TaskNotSetError("No task loaded or task has no test cases.")
        if not (0 <= self.current_test_input_idx < self.num_test_inputs_in_task):
            raise ARCError(f"Current test input index {self.current_test_input_idx} is out of bounds "
                           f"for {self.num_test_inputs_in_task} test inputs in task '{self.current_task_id}'.")

        test_pair = self.current_task_data.test[self.current_test_input_idx]
        return test_pair["input"], test_pair["output"]

    def _initialize_episode_state(self) -> None:
        """Sets up the environment state for the start of a new episode (for the current task/test_input)."""
        if not self.current_task_data:
            raise TaskNotSetError("Cannot initialize episode: No task has been set. Call set_task() first.")

        test_input_grid, self.target_grid = self._get_current_task_test_pair()

        # Initialize current_grid: usually starts as a copy of the test input,
        # or an empty canvas, depending on ARC problem philosophy.
        # Common: agent starts with the test input grid and modifies it.
        # Or, agent is GIVEN test input and an EMPTY canvas to draw on.
        # Let's assume agent modifies a copy of the test input grid.
        # This needs to fit self.env_config.canvas_size.
        # If test_input_grid is smaller, it might be padded or placed.
        # If larger, it might be cropped or error.
        # For now, assume test_input_grid fits or is the canvas.
        # A robust impl needs padding/cropping logic based on canvas_size.

        # Simple copy if shapes match, otherwise needs handling:
        if test_input_grid.shape == (self.env_config.canvas_size, self.env_config.canvas_size):
            self.current_grid = np.copy(test_input_grid)
        else:
            # Simple placement on canvas for now if smaller, error if larger
            self.current_grid = np.full(
                (self.env_config.canvas_size, self.env_config.canvas_size),
                ARC_BLACK, # Fill with black (or a configured background color)
                dtype=np.uint8
            )
            h, w = test_input_grid.shape
            if h > self.env_config.canvas_size or w > self.env_config.canvas_size:
                # For now, just log a warning and clip. A better solution is needed.
                print(f"Warning: Task '{self.current_task_id}' test input grid shape ({h},{w}) "
                      f"is larger than canvas ({self.env_config.canvas_size},{self.env_config.canvas_size}). Clipping.")
                h = min(h, self.env_config.canvas_size)
                w = min(w, self.env_config.canvas_size)
                self.current_grid[:h, :w] = test_input_grid[:h, :w]
            else: # Place smaller grid on canvas (top-left)
                self.current_grid[:h, :w] = test_input_grid

        self.current_selection_mask = None # Reset selection
        self.num_steps_taken = 0
        self.last_action_decoded = None
        self.last_reward = 0.0


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ARCStructuredObservation, Dict[str, Any]]:
        super().reset(seed=seed) # Handles RNG seeding

        # Options could specify a new task_id or test_input_idx to load
        if options:
            if "task_id" in options and options["task_id"] != self.current_task_id:
                self.set_task(options["task_id"])
                # current_test_input_idx is reset to 0 by set_task
            if "test_input_idx" in options:
                new_idx = options["test_input_idx"]
                if not self.current_task_data: raise TaskNotSetError("Cannot set test_input_idx: no task loaded.")
                if 0 <= new_idx < self.num_test_inputs_in_task:
                    self.current_test_input_idx = new_idx
                else:
                    raise ValueError(f"Invalid test_input_idx {new_idx} for task with {self.num_test_inputs_in_task} test inputs.")

        if not self.current_task_id or not self.current_task_data:
            # Try to load the first available task if none is set
            available_tasks = self.data_loader.list_available_tasks()
            if not available_tasks:
                raise TaskNotSetError("ARCEnv.reset() called but no task is set and no tasks are available from data loader.")
            print(f"Warning: No task set in ARCEnv.reset(). Loading first available task: {available_tasks[0]}")
            self.set_task(available_tasks[0])

        self._initialize_episode_state()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info


    def _apply_action_to_grid(self, decoded_ops: DecodedArcOps) -> Tuple[np.ndarray, bool]:
        """
        Applies the sequence of decoded operations to the current_grid.
        Returns the new grid and a boolean indicating if any change occurred.
        This is where DSL operations would interact with self.current_grid and self.current_selection_mask.
        """
        grid_before_action = np.copy(self.current_grid)
        temp_grid = np.copy(self.current_grid)

        # The DecodedArcOps is a tuple, e.g. (selection_op, color_op, transform_op)
        # The order of application matters. Assume selection -> color -> transform for now.
        # This sequence needs to be defined by the action space structure.
        # For ARCActionSpace, self.action_space._category_keys gives the order.

        active_selection_mask = self.current_selection_mask # Start with existing mask

        for op_idx, cat_key in enumerate(self.action_space._category_keys):
            if op_idx >= len(decoded_ops): break # Should not happen if lengths match

            op: Optional[BaseOperation] = decoded_ops[op_idx]
            if op is None: # No operation for this category in the action
                continue

            # How selection ops interact needs careful design.
            # If op is a "selection" type operation:
            if cat_key == "selection": # This relies on category names
                # Selection ops usually generate a new mask rather than modify the grid.
                # We need to access this generated mask.
                if hasattr(op, 'generate_mask') and callable(op.generate_mask):
                    active_selection_mask = op.generate_mask(temp_grid, existing_mask=active_selection_mask)
                    # The grid itself is not changed by the selection op directly.
                else:
                    print(f"Warning: Operation {op.to_string()} in category '{cat_key}' does not have 'generate_mask'. Cannot update selection.")
            else: # For non-selection operations (color, transform, etc.)
                temp_grid = op.apply(temp_grid, selection_mask=active_selection_mask)

        self.current_selection_mask = active_selection_mask # Persist the final selection mask for next step

        changed = not np.array_equal(grid_before_action, temp_grid)
        return temp_grid, changed


    def step(
        self, action: Union[int, Dict[str, int]]
    ) -> Tuple[ARCStructuredObservation, float, bool, bool, Dict[str, Any]]:
        if not self.current_task_data or self.target_grid is None:
            raise TaskNotSetError("Environment not reset or no task loaded. Call reset() before step().")

        self.num_steps_taken += 1

        # 1. Decode the action into DSL operation(s)
        try:
            decoded_ops: DecodedArcOps = self.action_space.decode(action)
            self.last_action_decoded = decoded_ops
        except Exception as e:
            # Invalid action format or decoding error
            # This should ideally be caught by action_space.contains() before step if agent checks.
            # If it happens here, it's a severe error.
            print(f"Error decoding action {action}: {e}")
            # Penalize heavily and end episode? Or return current state with large penalty?
            obs = self._get_observation() # Current observation
            info = self._get_info()
            info["error"] = f"Action decoding failed: {e}"
            info["invalid_action_penalty_applied"] = True
            # Terminate or truncate? Let's truncate.
            return obs, self.env_config.trunc_penalty, False, True, info


        # 2. Apply operation(s) to self.current_grid
        new_grid, changed_grid = self._apply_action_to_grid(decoded_ops)
        self.current_grid = new_grid # Update state

        # 3. Calculate reward
        reward = 0.0
        terminated = False

        # Penalty for each step
        reward += self.env_config.step_penalty

        # Penalty for no change (if configured)
        if not changed_grid and self.env_config.no_change_penalty != 0:
            reward += self.env_config.no_change_penalty
            # (info dict can signal this)

        # Check for solution
        if np.array_equal(self.current_grid, self.target_grid):
            terminated = True
            reward += self.env_config.completion_bonus
            # Potentially advance self.current_test_input_idx here if task has multiple test inputs
            # and we want to solve them sequentially within one "episode" concept.
            # Or, curriculum wrapper / outer loop handles moving to next test input via reset options.
            # For now, solving one test input means episode termination.
        else:
            # Optional: Penalty for incorrect shape if intermediate checks are desired
            # This is usually only checked at termination by comparing to target_grid.shape.
            # if self.current_grid.shape != self.target_grid.shape:
            #    reward += self.env_config.shape_penalty
            pass

        # 4. Check for truncation (max steps)
        truncated = False
        if self.env_config.max_steps is not None and self.num_steps_taken >= self.env_config.max_steps:
            truncated = True
            if not terminated: # Only apply trunc_penalty if not already solved
                 reward += self.env_config.trunc_penalty

        self.last_reward = reward

        # 5. Get new observation and info
        obs = self._get_observation()
        info = self._get_info()
        info["action_decoded"] = [op.to_string() if op else "NoOp" for op in decoded_ops]
        info["grid_changed"] = changed_grid
        if not changed_grid: info["no_op_occurred"] = True # Signal for wrappers

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> ARCStructuredObservation:
        if not self.current_task_data or self.target_grid is None: # Should be caught by callers
            raise TaskNotSetError("Cannot get observation: No task set or episode not initialized.")

        current_test_input, _ = self._get_current_task_test_pair()

        # Prepare train_pairs for observation space (padding if necessary)
        # ARCObservationSpace expects a fixed number of train pairs.
        obs_train_pairs = []
        num_actual_train_pairs = len(self.current_task_data.train)
        max_pairs_for_obs = self.observation_space.max_train_pairs

        for i in range(max_pairs_for_obs):
            if i < num_actual_train_pairs:
                pair = self.current_task_data.train[i]
                # Ensure grids fit canvas_size (pad/crop if needed)
                # For now, assume they are used as-is or ARCObservationSpace handles size.
                # ARCObservationSpace's Box spaces define the shape. Env must provide conforming arrays.
                # This means input/output grids from tasks should be padded/cropped here to fit.
                # This is a complex step omitted for brevity, assume loader provides correctly sized or
                # that ARCObservationSpace would need to be more dynamic (not good for fixed spaces).
                # Let's use a helper for padding/cropping.
                obs_train_pairs.append({
                    "input": self._prepare_grid_for_obs(pair["input"], "train_input"),
                    "output": self._prepare_grid_for_obs(pair["output"], "train_output")
                })
            else: # Pad with empty/dummy grids
                dummy_grid = np.full(
                    (self.env_config.canvas_size, self.env_config.canvas_size),
                    0, # Fill value for padding, could be special like -1 if space allows
                    dtype=np.uint8
                )
                obs_train_pairs.append({"input": dummy_grid, "output": np.copy(dummy_grid)})

        raw_obs_dict = {
            "task_grid": np.copy(self.current_grid), # Agent's current working grid
            "train_pairs": tuple(obs_train_pairs), # Must be tuple for gym.spaces.Tuple
            "test_input_grid": self._prepare_grid_for_obs(current_test_input, "test_input"),
            # Optional:
            # "current_selection_mask": np.copy(self.current_selection_mask) if self.current_selection_mask is not None \
            #                           else np.zeros_like(self.current_grid, dtype=bool),
        }
        return self.observation_space.structure_observation(raw_obs_dict)

    def _prepare_grid_for_obs(self, grid_data: np.ndarray, grid_name_for_log: str) -> np.ndarray:
        """Pads or crops a grid to fit the standard canvas size for observation."""
        target_shape = (self.env_config.canvas_size, self.env_config.canvas_size)
        if grid_data.shape == target_shape:
            return np.copy(grid_data)

        # Create a canvas of the target shape, filled with a background color (e.g., ARC_BLACK)
        canvas = np.full(target_shape, ARC_BLACK, dtype=grid_data.dtype)

        src_h, src_w = grid_data.shape
        dst_h, dst_w = target_shape

        # Determine copy region (clip if source is larger than destination)
        copy_h = min(src_h, dst_h)
        copy_w = min(src_w, dst_w)

        canvas[:copy_h, :copy_w] = grid_data[:copy_h, :copy_w]

        if src_h > dst_h or src_w > dst_w:
            print(f"Warning: Grid '{grid_name_for_log}' (shape {grid_data.shape}) was cropped to fit canvas shape {target_shape}.")
        # No warning for padding smaller grids, as that's expected.

        return canvas


    def _get_info(self) -> Dict[str, Any]:
        """Returns auxiliary information about the current state."""
        return {
            "task_id": self.current_task_id,
            "current_test_input_index": self.current_test_input_idx,
            "num_test_inputs_in_task": self.num_test_inputs_in_task,
            "steps_taken": self.num_steps_taken,
            "max_steps": self.env_config.max_steps,
            "last_action_decoded_str": [op.to_string() if op else "NoOp" for op in self.last_action_decoded] if self.last_action_decoded else None,
            "last_reward": self.last_reward,
            # "current_selection_mask": np.copy(self.current_selection_mask) if self.current_selection_mask is not None else None,
        }

    def render(self) -> Union[np.ndarray, None]:
        # Based on self.render_mode (set in __init__ from env_config)
        if self.render_mode == "human":
            # TODO: Implement human-friendly rendering (e.g., Pygame, Matplotlib)
            # For now, print to console as a fallback for human mode.
            self._render_ansi()
            return None
        elif self.render_mode == "rgb_array":
            # TODO: Convert self.current_grid to an RGB numpy array
            # This needs a color map from ARC int colors to RGB tuples.
            # Example: map {0:(0,0,0), 1:(0,0,255), ...} then create HxWx3 array.
            # Placeholder: return a grayscale representation if no color map yet.
            # Ensure it matches metadata["render_fps"] if that's relevant for rgb_array sequence.

            # Simple grayscale for now ( H x W ) -> ( H x W x 1 ) -> ( H x W x 3 )
            # Scale 0-9 to 0-255
            scaled_grid = (self.current_grid * (255/9.0)).astype(np.uint8)
            # If you need HxWx3:
            # return np.stack([scaled_grid]*3, axis=-1)
            # For now, let's return a simple upscaled version that might be displayable
            # This is a placeholder, a proper color mapping is needed.
            # Pygame display usually expects HxWx3 or WxHx3.
            # Matplotlib can handle HxW with a colormap.
            # Return a 3-channel RGB image.
            h, w = self.current_grid.shape
            rgb_grid = np.zeros((h, w, 3), dtype=np.uint8)
            # Basic color mapping (placeholder - use color_utils for better map)
            # 0:Black, 1:Blue, 2:Red, 3:Green, 4:Yellow, 5:Grey, 6:Magenta, 7:Orange, 8:Cyan, 9:Brown
            color_map = {
                0: [0,0,0], 1:[0,0,200], 2:[200,0,0], 3:[0,200,0], 4:[200,200,0],
                5: [128,128,128], 6:[200,0,200], 7:[255,165,0], 8:[0,200,200], 9:[139,69,19]
            }
            for r in range(h):
                for c in range(w):
                    rgb_grid[r,c,:] = color_map.get(self.current_grid[r,c], [255,255,255]) # Default white for unknown
            return rgb_grid

        elif self.render_mode == "ansi":
            return self._render_ansi()

        elif self.render_mode is None: # No rendering
            return None
        else:
            return super().render() # Let gym.Env handle unknown modes (raises error)

    def _render_ansi(self) -> str:
        # ANSI rendering: print grid to console with colors (if terminal supports)
        # This is a simplified version. Real ANSI might use special chars for colors.
        # For now, just pretty print the grid.
        if self.current_grid is None: return "Grid not initialized."

        # Try to use richer characters for display.
        # Unicode block characters: ▇ (U+2587), █ (U+2588)
        # Or just print numbers with padding.
        s = "Current Grid State:\n"
        for r in range(self.current_grid.shape[0]):
            s += "".join([f"{self.current_grid[r,c]:2}" for c in range(self.current_grid.shape[1])]) + "\n"

        if self.target_grid is not None:
            s += "\nTarget Grid State:\n"
            for r in range(self.target_grid.shape[0]):
                 s += "".join([f"{self.target_grid[r,c]:2}" for c in range(self.target_grid.shape[1])]) + "\n"

        if self.current_selection_mask is not None:
            s += "\nCurrent Selection Mask (1=Selected):\n"
            for r in range(self.current_selection_mask.shape[0]):
                 s += "".join([f"{int(self.current_selection_mask[r,c]):2}" for c in range(self.current_selection_mask.shape[1])]) + "\n"

        print(s) # For "human" mode if it falls back here.
        return s # For "ansi" mode.

    def close(self) -> None:
        # Perform any necessary cleanup (e.g., closing rendering windows)
        print("ARCEnv closed.")
        pass

    def get_challenge_info(self) -> Dict[str, Any]:
        """Returns information about the currently loaded ARC challenge/task."""
        if not self.current_task_data:
            return {"error": "No task currently loaded."}

        # Basic info from the task data itself
        info = {
            "task_id": self.current_task_id,
            "num_train_pairs": len(self.current_task_data.train),
            "num_test_inputs": self.num_test_inputs_in_task,
            "current_test_input_index": self.current_test_input_idx,
        }
        # Potentially add more details if ARCTaskData has them (e.g. from a ChallengeDataset)
        if hasattr(self.current_task_data, 'metadata') and self.current_task_data.metadata: # type: ignore
            info.update(self.current_task_data.metadata) # type: ignore

        return info

    # TODO: Implement helper methods if needed:
    # - _is_task_solved()
    # - _calculate_reward_components()
    # - Methods for interacting with DSL operations if env directly manages them.

    # Property to access unwrapped env easily if needed (standard gym pattern)
    @property
    def unwrapped(self) -> 'ARCEnv':
        return self
```

This is a substantial initial implementation. It includes:
*   Initialization with configurations, data loader, and operation registry.
*   Core gym methods: `reset`, `step`, `render`, `close`.
*   ARC-specific methods: `set_task`, `get_challenge_info`.
*   Internal logic for managing task state, test input indexing, observations, and basic reward calculation.
*   Placeholders for rendering and more complex DSL interaction / grid operations.

There's a lot more detail that could be added, especially around robust grid handling (padding/cropping for observations), DSL execution nuances (how selection masks are precisely managed and passed between operations), and sophisticated rendering. But this forms a solid base.

I'll now create placeholder files for `variants/mini_arc.py` and `variants/multi_task.py`.
