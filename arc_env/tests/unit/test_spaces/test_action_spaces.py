import pytest
import numpy as np
from gymnasium.spaces import Discrete, Dict as GymDict

from arc_env.spaces.action_spaces import ARCActionSpace
from arc_env.config.action_space import ActionSpaceConfig
# from arc_env.dsl.core.operation_registry import OperationRegistry # From fixture
# from arc_env.dsl.core.base_operations import BaseOperation # For type checks
# from arc_env.dsl.presets.default import DefaultPresetLoader # Loaded in fixture
# from arc_env.dsl.presets.minimal import MinimalPresetLoader # For testing other presets

# Uses fixtures from conftest.py:
# - dummy_operation_registry (pre-loaded with "default" preset and its ops)
# - dummy_action_space_config (uses "default" preset)

def test_arc_action_space_creation_factorized(dummy_operation_registry, dummy_action_space_config):
    """Test creation of ARCActionSpace in 'factorized' mode."""
    config = dummy_action_space_config
    config.mode = "factorized" # Ensure mode

    action_space = ARCActionSpace(config, dummy_operation_registry)

    assert isinstance(action_space, ARCActionSpace)
    assert isinstance(action_space.underlying_gym_space, GymDict)

    # Check if categories from default preset are in the Dict space
    # (e.g., "selection", "color", "transform" are typical)
    default_preset_ops = dummy_operation_registry.get_operations_for_preset("default")
    assert isinstance(default_preset_ops, dict)

    expected_categories_in_space = [
        cat for cat, ops_list in default_preset_ops.items() if ops_list # Only categories with ops
    ]
    # The order in ARCActionSpace._category_keys might differ from dict iteration order.
    # Check that all expected categories are present.
    for cat_key in expected_categories_in_space:
        assert cat_key in action_space.underlying_gym_space.spaces
        assert isinstance(action_space.underlying_gym_space.spaces[cat_key], Discrete)
        assert action_space.underlying_gym_space.spaces[cat_key].n == len(default_preset_ops[cat_key])


def test_arc_action_space_creation_joint(dummy_operation_registry, dummy_action_space_config):
    """Test creation of ARCActionSpace in 'joint' mode."""
    config = dummy_action_space_config
    config.mode = "joint"

    action_space = ARCActionSpace(config, dummy_operation_registry)

    assert isinstance(action_space, ARCActionSpace)
    assert isinstance(action_space.underlying_gym_space, Discrete)

    # Calculate expected total size for joint space based on "default" preset
    default_preset_ops = dummy_operation_registry.get_operations_for_preset("default")
    assert isinstance(default_preset_ops, dict)

    # Use the category order defined in ARCActionSpace._category_keys for calculation
    # This order is based on a preferred_order then dict items.
    # For "default" preset, this likely includes "selection", "color", "transform".

    cat_sizes = []
    # This relies on knowing the internal ordered _category_keys of the instance,
    # or re-implementing its logic for category selection and ordering.
    # For simplicity, let's assume the main categories are present from default preset.
    # A more robust test would inspect action_space._category_keys directly.

    # Replicate category selection logic from ARCActionSpace to predict size:
    preferred_order = ["selection", "color", "transform"]
    temp_categories_for_size_calc: Dict[str, List] = {}
    for cat_name in preferred_order:
        if cat_name in default_preset_ops and default_preset_ops[cat_name]:
            temp_categories_for_size_calc[cat_name] = default_preset_ops[cat_name]
    for cat_name, ops_list in default_preset_ops.items():
        if cat_name not in temp_categories_for_size_calc and ops_list:
            temp_categories_for_size_calc[cat_name] = ops_list

    if not temp_categories_for_size_calc:
        pytest.fail("No operation categories found for joint space size calculation.")

    expected_total_size = 1
    for cat_key_ordered in temp_categories_for_size_calc.keys(): # Order matters for some interpretations
        ops_in_cat = temp_categories_for_size_calc[cat_key_ordered]
        if ops_in_cat: # Should always be true due to filtering
            expected_total_size *= len(ops_in_cat)

    assert action_space.underlying_gym_space.n == expected_total_size


def test_arc_action_space_sample_and_decode_factorized(dummy_operation_registry, dummy_action_space_config):
    """Test sampling and decoding for 'factorized' mode."""
    config = dummy_action_space_config
    config.mode = "factorized"
    action_space = ARCActionSpace(config, dummy_operation_registry)

    # Sample a raw action
    raw_action = action_space.sample()
    assert isinstance(raw_action, dict) # Factorized sample is a dict
    assert action_space.contains(raw_action)

    # Decode the action
    decoded_ops_tuple = action_space.decode(raw_action)
    assert isinstance(decoded_ops_tuple, tuple)
    # DecodedArcOps is Tuple[Optional[BaseOp], Optional[BaseOp], Optional[BaseOp]]
    assert len(decoded_ops_tuple) == 3

    from arc_env.dsl.core.base_operations import BaseOperation
    num_valid_ops = 0
    for i, cat_key in enumerate(action_space._category_keys):
        if i >= 3 : break # Max 3 ops in DecodedArcOps tuple
        op_instance = decoded_ops_tuple[i]
        if cat_key in raw_action: # If this category was part of the action
            assert op_instance is not None
            assert isinstance(op_instance, BaseOperation)
            # Check if it's the correct op from the category list
            expected_op = action_space._op_categories[cat_key][raw_action[cat_key]]
            assert op_instance is expected_op # Should be the same instance
            num_valid_ops +=1
        # else: op_instance should be None if category not in raw_action (not typical for factorized all-cats)

    assert num_valid_ops == len(action_space._category_keys), "Mismatch in number of decoded ops vs categories in action"


def test_arc_action_space_sample_and_decode_joint(dummy_operation_registry, dummy_action_space_config):
    """Test sampling and decoding for 'joint' mode."""
    config = dummy_action_space_config
    config.mode = "joint"
    action_space = ARCActionSpace(config, dummy_operation_registry)

    raw_action = action_space.sample() # integer
    assert isinstance(raw_action, (int, np.integer))
    assert action_space.contains(raw_action)

    decoded_ops_tuple = action_space.decode(raw_action)
    assert isinstance(decoded_ops_tuple, tuple)
    assert len(decoded_ops_tuple) == 3

    from arc_env.dsl.core.base_operations import BaseOperation
    num_decoded_categories = min(len(action_space._category_keys), 3)

    for i in range(num_decoded_categories):
        op_instance = decoded_ops_tuple[i]
        assert op_instance is not None, f"Decoded op at index {i} should not be None for joint space."
        assert isinstance(op_instance, BaseOperation)

    # Test that all ops are from the expected categories
    # This requires reconstructing the indices from the raw_action, which is complex.
    # A simpler check: ensure the types of ops match what's in the registry for those categories.
    # More thorough: encode the decoded_ops_tuple back and see if it matches raw_action (if encoder is available).


def test_arc_action_space_with_minimal_preset(dummy_operation_registry, dummy_action_space_config):
    """Test action space with a different preset like "minimal"."""
    # Ensure "minimal" preset and its ops are loaded into the registry for this test
    from arc_env.dsl.presets.minimal import MinimalPresetLoader
    MinimalPresetLoader(dummy_operation_registry).load() # exist_ok=True in loader

    config = ActionSpaceConfig(mode="factorized", preset="minimal")
    if "minimal" not in config.available_presets: config.available_presets.append("minimal")

    action_space = ARCActionSpace(config, dummy_operation_registry)

    assert isinstance(action_space.underlying_gym_space, GymDict)

    minimal_preset_ops = dummy_operation_registry.get_operations_for_preset("minimal")
    assert isinstance(minimal_preset_ops, dict)

    expected_minimal_categories = [cat for cat, ops_list in minimal_preset_ops.items() if ops_list]
    assert len(action_space.underlying_gym_space.spaces) == len(expected_minimal_categories)

    for cat_key in expected_minimal_categories:
        assert cat_key in action_space.underlying_gym_space.spaces
        assert len(minimal_preset_ops[cat_key]) == action_space.underlying_gym_space.spaces[cat_key].n


def test_arc_action_space_invalid_config(dummy_operation_registry):
    """Test creation with invalid configurations."""
    # Invalid mode
    with pytest.raises(ConfigurationError): # Or ValueError depending on ActionSpaceConfig validation
        ActionSpaceConfig(mode="invalid_mode_string") # type: ignore

    # Preset not found in registry (if registry is strict and preset not loaded)
    # The dummy_operation_registry loads "default". If we ask for "non_existent_preset":
    # ARCActionSpace itself might raise ConfigurationError or ARCError during _build_operations_from_preset
    config_bad_preset = ActionSpaceConfig(preset="non_existent_preset_123")
    with pytest.raises(ConfigurationError): # Or ARCError from registry
        ARCActionSpace(config_bad_preset, dummy_operation_registry)

# Add more tests:
# - Specific decoding logic for joint space (provide an int, check the exact ops).
# - Edge cases: empty categories in preset, presets with more than 3 categories for joint mode.
# - Interaction with ActionSpaceConfig's `allow_noop` if that feature is implemented.
