from __future__ import annotations

from typing import Optional, Dict, Any

from arc_env.environments.arc_env import ARCEnv
from arc_env.config.environment import EnvironmentConfig
from arc_env.config.action_space import ActionSpaceConfig
from arc_env.data.loaders.base import BaseDataLoader # For type hint
from arc_env.dsl.core.operation_registry import OperationRegistry # For type hint
# Potentially, import a minimal preset loader if MiniARC uses a specific minimal set of ops
# from arc_env.dsl.presets.minimal import MinimalPresetLoader

class MiniARCEnv(ARCEnv):
    """
    A simplified variant of the main ARCEnv.
    This could feature:
    - Smaller default canvas size.
    - A restricted set of DSL operations (e.g., using the "minimal" preset).
    - Fewer colors.
    - Simpler tasks or a specific subset of tasks.
    """

    def __init__(
        self,
        env_config: Optional[EnvironmentConfig] = None,
        action_space_config: Optional[ActionSpaceConfig] = None,
        data_loader: Optional[BaseDataLoader] = None,
        operation_registry: Optional[OperationRegistry] = None,
        initial_task_id: Optional[str] = None
    ):
        # --- Setup MiniARC specific configurations ---
        # If no specific configs are provided, create defaults tailored for MiniARC.

        # Default EnvironmentConfig for MiniARC (e.g., smaller canvas)
        mini_env_cfg = EnvironmentConfig(
            canvas_size=10, # Smaller canvas
            max_steps=50    # Fewer steps
            # Other parameters can be inherited from EnvironmentConfig defaults or set here
        )
        if env_config: # If user provides one, merge or override
            mini_env_cfg = mini_env_cfg.merge(env_config) # type: ignore # BaseConfig merge

        # Default ActionSpaceConfig for MiniARC (e.g., "minimal" preset)
        mini_action_cfg = ActionSpaceConfig(
            preset="minimal" # Use a predefined minimal set of operations
            # Potentially ensure available_presets in config includes "minimal"
        )
        if "minimal" not in mini_action_cfg.available_presets: # Ensure it's listed
             mini_action_cfg.available_presets.append("minimal")

        if action_space_config:
            mini_action_cfg = mini_action_cfg.merge(action_space_config) # type: ignore

        # Operation Registry: Ensure minimal preset is loaded if not already
        # This is a bit tricky if the global op_registry is passed around.
        # For MiniARC, we might want to ensure its specific preset is primary.
        # If a registry is passed, we assume it's already populated.
        # If not, we create one and load minimal ops.
        registry_to_use = operation_registry
        if registry_to_use is None:
            registry_to_use = OperationRegistry()
            # Load minimal preset into this new registry
            try:
                from arc_env.dsl.presets.minimal import MinimalPresetLoader
                MinimalPresetLoader(registry_to_use).load()
                print("MiniARCEnv: Loaded MinimalPreset into new OperationRegistry.")
            except ImportError:
                print("Warning: MiniARCEnv could not load MinimalPresetLoader. DSL operations may be limited.")
        else: # Registry provided, check if "minimal" preset exists, if not, try to load.
            if "minimal" not in registry_to_use.list_available_presets():
                print(f"Warning: 'minimal' preset not found in provided OperationRegistry for MiniARCEnv. Attempting to load.")
                try:
                    from arc_env.dsl.presets.minimal import MinimalPresetLoader
                    MinimalPresetLoader(registry_to_use).load() # exist_ok=True in loader
                except Exception as e:
                    print(f"Warning: Failed to load 'minimal' preset into existing registry: {e}")


        # Data Loader: Could be specific to simpler tasks, or use the same as ARCEnv.
        # For now, use the same logic as ARCEnv for data_loader.

        super().__init__(
            env_config=mini_env_cfg,
            action_space_config=mini_action_cfg,
            data_loader=data_loader, # Let super handle default loading if None
            operation_registry=registry_to_use,
            initial_task_id=initial_task_id
        )

        print(f"MiniARCEnv initialized. Canvas: {self.env_config.canvas_size}x{self.env_config.canvas_size}, "
              f"Action Preset: '{self.action_space_config.preset}'.")

    # MiniARCEnv might override some methods if its behavior differs significantly,
    # but for many cases, just configuring the parent ARCEnv differently is enough.
    # For example, if MiniARC tasks always have exactly one test input:
    # def _get_current_task_test_pair(self) -> Tuple[np.ndarray, np.ndarray]:
    #     # Override if MiniARC has a different assumption about test pairs
    #     if self.current_task_data and len(self.current_task_data.test) != 1:
    #         # This is an assertion about the data expected by MiniARC
    #         # print(f"Warning: MiniARC task '{self.current_task_id}' has "
    #         #       f"{len(self.current_task_data.test)} test pairs, expected 1.")
    #         pass # Proceed with parent logic for now
    #     return super()._get_current_task_test_pair()


# Example Usage:
# if __name__ == "__main__":
#     # This example assumes you have dummy task files loadable by ARCFileLoader
#     # and that the "minimal" preset's operations are defined.
#     from pathlib import Path
#     import json
#     import shutil

#     # Setup dummy data for the loader (as in ARCEnv example)
#     dummy_mini_task_dir = Path("temp_mini_arc_tasks")
#     dummy_mini_task_dir.mkdir(exist_ok=True)
#     task_content = {
#         "mini_task_01": {"train": [{"input": [[1]], "output": [[0]]}], "test": [{"input": [[1]], "output": [[0]]}]}
#     }
#     with open(dummy_mini_task_dir / "mini_task_01.json", "w") as f: json.dump(task_content["mini_task_01"], f)

#     # Create a specific env_config for the MiniARC instance if needed for data loading path
#     cfg = EnvironmentConfig(data_path=str(dummy_mini_task_dir))

#     try:
#         print("--- Initializing MiniARCEnv ---")
#         # Note: MiniARCEnv will try to load "minimal" preset.
#         # Ensure its operations (FillOperation, SelectByColorOperation, RotateOperation) are defined.
#         # The example operations in dsl/operations/* should cover these.
#         mini_env = MiniARCEnv(env_config=cfg, initial_task_id="mini_task_01")

#         print(f"\nMiniARCEnv Action Space uses preset: '{mini_env.action_space.config.preset}'") # Should be 'minimal'
#         print(f"MiniARCEnv Canvas Size: {mini_env.env_config.canvas_size}") # Should be 10 (MiniARC default)

#         obs, info = mini_env.reset()
#         print("\nMiniARCEnv reset successful.")
#         print(f"Task ID: {info.get('task_id')}, Initial Observation (task_grid sample):")
#         print(obs["task_grid"][0:3, 0:3]) # Print top-left corner

#         # Try a step with a sample action from its (likely minimal) action space
#         action = mini_env.action_space.sample()
#         obs, reward, terminated, truncated, info = mini_env.step(action)
#         print("\nMiniARCEnv step successful.")
#         print(f"Action taken (decoded): {info.get('action_decoded_str')}")
#         print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

#     except Exception as e:
#         print(f"Error during MiniARCEnv example: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         if dummy_mini_task_dir.exists():
#             shutil.rmtree(dummy_mini_task_dir)
