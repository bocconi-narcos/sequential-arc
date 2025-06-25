from __future__ import annotations # For Type hinting OperationRegistry if it were in the same file etc.

from typing import Dict, Any, Union, Tuple, Callable, List, cast, Optional # Added Optional
import gymnasium as gym
import numpy as np

from arc_env.core.base_action_space import BaseActionSpace # DecodedActionType is generic here
from arc_env.config.action_space import ActionSpaceConfig
from arc_env.dsl.core.operation_registry import OperationRegistry
from arc_env.dsl.core.base_operations import BaseOperation
from arc_env.exceptions import ConfigurationError, ARCError

# Define a more specific type for the decoded action if possible.
# For this example, it's a tuple of up to three operations (or callables).
# If an action only involves one operation type, others can be None or a placeholder.
DecodedArcOps = Tuple[Optional[BaseOperation], Optional[BaseOperation], Optional[BaseOperation]]


class ARCActionSpace(BaseActionSpace[DecodedArcOps]):
    """
    Improved ARC action space with better modularity, configured via ActionSpaceConfig
    and using an OperationRegistry.

    This action space can operate in two modes:
    1.  "factorized": The gym space is a Dict space, with separate entries for
        different operation categories (e.g., color, selection, transform).
        The agent picks one operation from each category (or a subset).
    2.  "joint": The gym space is a single Discrete space, where each integer
        maps to a unique combination of operations from different categories.

    The `decode` method translates the raw gym action into a tuple of
    instantiated `BaseOperation` objects.
    """

    def __init__(self, config: ActionSpaceConfig, operation_registry: OperationRegistry) -> None:
        super().__init__() # BaseActionSpace init

        if not isinstance(config, ActionSpaceConfig):
            raise ConfigurationError("config must be an instance of ActionSpaceConfig.")
        self.config = config

        if not isinstance(operation_registry, OperationRegistry):
            raise ARCError("operation_registry must be an instance of OperationRegistry.")
        self._registry = operation_registry

        # These will store lists of *instantiated* operations for the chosen preset.
        self._op_categories: Dict[str, List[BaseOperation]] = {}
        self._category_keys: List[str] = [] # Keep an ordered list of categories used

        self._build_operations_from_preset()
        self._build_underlying_gym_space()

    def _build_operations_from_preset(self) -> None:
        """
        Populates self._op_categories using the configured preset from the registry.
        Example categories: "color", "selection", "transform".
        The actual categories used will depend on the preset definition in the registry.
        """
        try:
            # Get all operations for the configured preset
            preset_ops = self._registry.get_operations_for_preset(self.config.preset)
            if not isinstance(preset_ops, dict):
                 raise ARCError(f"Expected a dictionary of operations from preset '{self.config.preset}', "
                                f"got {type(preset_ops)}. Ensure the preset is correctly registered.")

            # Filter for categories that have operations.
            # The order of categories can be important for joint space encoding/decoding.
            # For now, let's use a fixed preferred order if they exist, or dict order.
            # Preferred order for factorized/joint spaces:
            preferred_order = ["selection", "color", "transform"] # Example order

            temp_categories: Dict[str, List[BaseOperation]] = {}
            for cat_name in preferred_order:
                if cat_name in preset_ops and preset_ops[cat_name]:
                    temp_categories[cat_name] = preset_ops[cat_name]

            # Add any other categories from the preset not in preferred_order
            for cat_name, ops_list in preset_ops.items():
                if cat_name not in temp_categories and ops_list:
                    temp_categories[cat_name] = ops_list

            self._op_categories = temp_categories
            self._category_keys = list(self._op_categories.keys()) # Store the actual order used

            if not self._op_categories:
                raise ConfigurationError(
                    f"No operations found for preset '{self.config.preset}'. "
                    "The action space would be empty."
                )
            if len(self._category_keys) > 3 and self.config.mode == "joint":
                # The DecodedArcOps tuple is fixed at 3 elements.
                # If more categories, the joint space logic and DecodedArcOps need adjustment.
                print(f"Warning: Preset '{self.config.preset}' has {len(self._category_keys)} operation categories. "
                      f"The 'joint' mode's decode method currently assumes up to 3 categories for DecodedArcOps. "
                      "Factorized mode is more flexible with category counts.")


        except Exception as e: # Catch errors from registry (e.g., preset not found)
            raise ConfigurationError(f"Failed to build operations for preset '{self.config.preset}': {e}")

    @property
    def underlying_gym_space(self) -> gym.Space:
        return self._gym_space

    def _build_underlying_gym_space(self) -> None:
        """Build the underlying gym space based on the mode and loaded operations."""
        if not self._category_keys:
             raise ARCError("Cannot build gym space: No operation categories loaded. Check preset configuration.")

        if self.config.mode == "factorized":
            gym_space_dict: Dict[str, gym.Space] = {}
            for cat_key in self._category_keys:
                num_ops_in_cat = len(self._op_categories[cat_key])
                if num_ops_in_cat == 0:
                    # This case should ideally be filtered out by _build_operations_from_preset
                    # or raise an error there if a category ends up empty.
                    # If we allow categories with 0 ops, Discrete(0) is invalid.
                    # For now, assume num_ops_in_cat > 0 due to earlier checks.
                    print(f"Warning: Category '{cat_key}' has no operations for factorized space. Skipping.")
                    continue
                gym_space_dict[cat_key] = gym.spaces.Discrete(num_ops_in_cat)

            if not gym_space_dict:
                 raise ARCError("Cannot build factorized gym space: No categories with operations found.")
            self._gym_space = gym.spaces.Dict(gym_space_dict)

        elif self.config.mode == "joint":
            # Calculate total size for the joint Discrete space
            # This assumes agent performs one op from each category (or subset if some are optional - not handled yet)
            # The number of active categories for joint space is important.
            # Let's assume the first up to 3 categories from self._category_keys are used.

            cat_sizes = [len(self._op_categories[key]) for key in self._category_keys if self._op_categories[key]]
            if not cat_sizes:
                 raise ARCError("Cannot build joint gym space: No categories with operations for joint encoding.")

            total_size = 1
            for size in cat_sizes:
                total_size *= size

            if total_size == 0 : # Should not happen if cat_sizes is not empty and sizes are > 0
                raise ARCError("Calculated total size for joint action space is 0. Check operation categories.")

            self._gym_space = gym.spaces.Discrete(total_size)
        else:
            raise ConfigurationError(f"Unknown action space mode: {self.config.mode}")

    def decode(self, action: Union[int, Dict[str, int]]) -> DecodedArcOps:
        """
        Decode raw gym action to a tuple of BaseOperation instances.
        The tuple corresponds to (selection_op, color_op, transform_op) by default order.
        If categories are different, this mapping needs to be more flexible.
        """
        # Use self._category_keys to determine the order and number of operations.
        # The DecodedArcOps is (Op1, Op2, Op3). We need to map self._category_keys to these slots.
        # Example: If self._category_keys = ["selection", "color", "transform"]
        # op1_list = self._op_categories["selection"]
        # op2_list = self._op_categories["color"]
        # op3_list = self._op_categories["transform"]

        decoded_ops: List[Optional[BaseOperation]] = [None, None, None] # Max 3 ops for DecodedArcOps

        if self.config.mode == "factorized":
            if not isinstance(action, dict):
                raise ValueError(f"Factorized mode expects a dict action, got {type(action)}.")

            for i, cat_key in enumerate(self._category_keys):
                if i >= 3: break # Only fill up to 3 slots in DecodedArcOps
                if cat_key in action:
                    op_idx = action[cat_key]
                    if 0 <= op_idx < len(self._op_categories[cat_key]):
                        decoded_ops[i] = self._op_categories[cat_key][op_idx]
                    else:
                        raise ValueError(f"Invalid action index {op_idx} for category '{cat_key}'.")
                # else: if a category is missing in action dict, it's None (or could be an error)

        elif self.config.mode == "joint":
            if not isinstance(action, (int, np.integer)): # Check for numpy int types too
                raise ValueError(f"Joint mode expects an integer action, got {type(action)}.")

            # Perform joint decoding based on the order in self._category_keys
            # Example for 3 categories:
            # c1_size = len(self._op_categories[self._category_keys[0]])
            # c2_size = len(self._op_categories[self._category_keys[1]])
            # c3_size = len(self._op_categories[self._category_keys[2]])
            #
            # idx0 = action // (c2_size * c3_size)
            # idx1 = (action % (c2_size * c3_size)) // c3_size
            # idx2 = action % c3_size
            #
            # decoded_ops[0] = self._op_categories[self._category_keys[0]][idx0]
            # decoded_ops[1] = self._op_categories[self._category_keys[1]][idx1]
            # decoded_ops[2] = self._op_categories[self._category_keys[2]][idx2]

            current_action_val = int(action)
            # Iterate backwards through categories to correctly use modulo arithmetic for decoding
            # This is a general way to decode from a joint discrete space
            #
            # Product of sizes of remaining categories to decode
            # E.g., if sizes are [S1, S2, S3, S4], for S1, divisor is S2*S3*S4. For S2, S3*S4 etc.

            num_cats_to_decode = min(len(self._category_keys), 3) # Up to 3 ops

            for i in range(num_cats_to_decode):
                cat_key = self._category_keys[i]
                ops_list = self._op_categories[cat_key]
                current_cat_size = len(ops_list)
                if current_cat_size == 0: continue # Should not happen

                # Calculate product of sizes of subsequent categories
                divisor = 1
                for j in range(i + 1, num_cats_to_decode): # Only consider cats that will be decoded
                    next_cat_key = self._category_keys[j]
                    if self._op_categories[next_cat_key]: # Check if list is not empty
                         divisor *= len(self._op_categories[next_cat_key])

                op_idx = current_action_val // divisor
                if not (0 <= op_idx < current_cat_size):
                    raise ValueError(f"Decoded op_idx {op_idx} out of bounds for category '{cat_key}' size {current_cat_size}. Action: {action}")

                decoded_ops[i] = ops_list[op_idx]
                current_action_val %= divisor
        else:
            # Should be caught by constructor or config validation
            raise ARCError(f"Unsupported action space mode: {self.config.mode}")

        return cast(DecodedArcOps, tuple(decoded_ops))


    def __repr__(self) -> str:
        return (f"ARCActionSpace(config={self.config}, "
                f"underlying_space={self.underlying_gym_space}, "
                f"categories={self._category_keys})")


# Example Usage (Illustrative - requires OperationRegistry and ops to be set up)
# if __name__ == "__main__":
#     from arc_env.dsl.core.operation_registry import OperationRegistry
#     from arc_env.dsl.presets.default import DefaultPresetLoader # Assuming this registers ops and "default" preset
#     from arc_env.config.action_space import ActionSpaceConfig

#     # 1. Setup Registry and Load Presets
#     registry = OperationRegistry()
#     DefaultPresetLoader(registry).load() # Loads "default" preset and associated operations

#     # 2. Configure Action Space
#     # Factorized mode
#     factorized_config = ActionSpaceConfig(mode="factorized", preset="default")
#     factorized_action_space = ARCActionSpace(factorized_config, registry)
#     print("Factorized Action Space:", factorized_action_space)
#     print("Underlying Gym Space (Factorized):", factorized_action_space.underlying_gym_space)

#     sample_factorized_action = factorized_action_space.sample()
#     print("Sample Factorized Action (raw):", sample_factorized_action)
#     decoded_factorized = factorized_action_space.decode(sample_factorized_action)
#     print("Decoded Factorized Action:", [(op.to_string() if op else None) for op in decoded_factorized])

#     # Joint mode
#     joint_config = ActionSpaceConfig(mode="joint", preset="default")
#     joint_action_space = ARCActionSpace(joint_config, registry)
#     print("\nJoint Action Space:", joint_action_space)
#     print("Underlying Gym Space (Joint):", joint_action_space.underlying_gym_space)

#     sample_joint_action = joint_action_space.sample()
#     print("Sample Joint Action (raw):", sample_joint_action)
#     decoded_joint = joint_action_space.decode(sample_joint_action)
#     print("Decoded Joint Action:", [(op.to_string() if op else None) for op in decoded_joint])

#     # Test with a specific joint action (assuming 3 categories in 'default' preset)
#     # E.g., if cat sizes are [sel_size, color_size, transform_size]
#     # action = (idx_sel * color_size * transform_size) + (idx_color * transform_size) + idx_transform
#     # This depends on the actual number of ops in each category for the "default" preset.
#     # Let's assume default preset has selection:2, color:2, transform:2 for simplicity here.
#     # Then total size is 8. Action for (sel_op_0, color_op_1, transform_op_0)
#     # idx_sel=0, idx_color=1, idx_transform=0
#     # action_val = (0 * 2 * 2) + (1 * 2) + 0 = 2
#     # This requires knowing the exact structure of "default" preset from DefaultPresetLoader.
#     # The example ops in DefaultPresetLoader might give different counts.
#     # For default_preset_config in `default.py`:
#     # color: 2 ops, selection: 4 ops, transform: 2 ops
#     # If order is sel, color, transform: S=4, C=2, T=2. Total=16
#     # Action for (sel_op[1], color_op[0], transform_op[1]):
#     # action = (1 * 2 * 2) + (0 * 2) + 1 = 4 + 0 + 1 = 5
#     if joint_action_space.underlying_gym_space.contains(5):
#         print("\nDecoding joint action '5':")
#         decoded_5 = joint_action_space.decode(5)
#         print("Decoded action 5:", [(op.to_string() if op else None) for op in decoded_5])
#     else:
#         print(f"\nJoint action '5' is out of bounds for space size {joint_action_space.underlying_gym_space.n}")
