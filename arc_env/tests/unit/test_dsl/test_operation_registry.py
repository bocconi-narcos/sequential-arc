import pytest
from typing import Dict, List, Any, Type

from arc_env.dsl.core.operation_registry import OperationRegistry
from arc_env.dsl.core.base_operations import BaseOperation
from arc_env.exceptions import OperationError, TypeError as CustomTypeError

# --- Dummy Operations for Testing ---
class DummyColorOp(BaseOperation):
    def __init__(self, color: int = 0, **kwargs): super().__init__(color=color, **kwargs); self.color = color
    def apply(self, grid, selection_mask=None): return grid + self.color # Dummy effect
    def to_string(self): return f"DummyColorOp(color={self.color})"

class DummySelectOp(BaseOperation):
    def __init__(self, method: str = "all", **kwargs): super().__init__(method=method, **kwargs); self.method = method
    def apply(self, grid, selection_mask=None): return grid # No change to grid
    def generate_mask(self, grid, existing_mask=None): return grid > -1 # Selects all
    def to_string(self): return f"DummySelectOp(method='{self.method}')"

class NotAnOperation: # Not inheriting from BaseOperation
    pass

# --- Fixture for a fresh OperationRegistry ---
@pytest.fixture
def registry() -> OperationRegistry:
    """Provides a fresh OperationRegistry instance for each test."""
    return OperationRegistry()

# --- Tests for Operation Registration ---
def test_register_operation_success(registry: OperationRegistry):
    registry.register_operation("color", "dummy_red", DummyColorOp)
    op_class = registry.get_operation_class("color", "dummy_red")
    assert op_class is DummyColorOp
    assert "color" in registry.list_available_operations() # type: ignore # list_available_operations returns dict here
    assert "dummy_red" in registry.list_available_operations("color") # type: ignore # list_available_operations returns list here

def test_register_operation_already_exists_fail(registry: OperationRegistry):
    registry.register_operation("color", "dummy_op", DummyColorOp)
    with pytest.raises(OperationError, match="already registered"):
        registry.register_operation("color", "dummy_op", DummyColorOp, exist_ok=False)

def test_register_operation_already_exists_ok(registry: OperationRegistry):
    registry.register_operation("color", "dummy_op", DummyColorOp)
    # Should not raise error
    registry.register_operation("color", "dummy_op", DummySelectOp, exist_ok=True) # Overwrite with different class
    op_class = registry.get_operation_class("color", "dummy_op")
    assert op_class is DummySelectOp # Check it was overwritten

def test_register_operation_invalid_class_type(registry: OperationRegistry):
    with pytest.raises(CustomTypeError, match="must be a subclass of BaseOperation"): # Using our custom TypeError
        registry.register_operation("category1", "invalid_op", NotAnOperation) # type: ignore

# --- Tests for Preset Registration ---
def test_register_preset_success(registry: OperationRegistry):
    # Register operations first
    registry.register_operation("color", "op_c1", DummyColorOp)
    registry.register_operation("select", "op_s1", DummySelectOp)

    preset_config = {
        "color": [{"name": "op_c1", "kwargs": {"color": 5}}],
        "select": [{"name": "op_s1", "kwargs": {"method": "by_color"}}]
    }
    registry.register_preset("my_preset", preset_config)
    assert "my_preset" in registry.list_available_presets()

def test_register_preset_op_not_registered_fail(registry: OperationRegistry):
    preset_config_bad_op = {
        "color": [{"name": "non_existent_op", "kwargs": {}}]
    }
    with pytest.raises(OperationError, match="is not registered"):
        registry.register_preset("bad_preset", preset_config_bad_op)

def test_register_preset_already_exists_fail(registry: OperationRegistry):
    registry.register_operation("color", "op_c1", DummyColorOp) # Op for preset
    preset_config = {"color": [{"name": "op_c1", "kwargs": {}}]}
    registry.register_preset("existing_preset", preset_config)
    with pytest.raises(OperationError, match="already registered"):
        registry.register_preset("existing_preset", preset_config, exist_ok=False)

def test_register_preset_already_exists_ok(registry: OperationRegistry):
    registry.register_operation("color", "op_c1", DummyColorOp)
    preset_config1 = {"color": [{"name": "op_c1", "kwargs": {"color": 1}}]}
    registry.register_preset("preset_to_overwrite", preset_config1)

    preset_config2 = {"color": [{"name": "op_c1", "kwargs": {"color": 2}}]} # Different kwargs
    registry.register_preset("preset_to_overwrite", preset_config2, exist_ok=True) # Overwrite

    # Verify overwrite by checking instantiated op's params
    ops = registry.get_operations("color", "preset_to_overwrite")
    assert len(ops) == 1
    assert isinstance(ops[0], DummyColorOp)
    assert ops[0].params.get("color") == 2


# --- Tests for Retrieving Operations ---
def test_get_operations_for_preset_specific_category(registry: OperationRegistry):
    registry.register_operation("color", "c_op", DummyColorOp)
    registry.register_operation("select", "s_op", DummySelectOp)
    preset_cfg = {
        "color": [{"name": "c_op", "kwargs": {"color": 3}}],
        "select": [{"name": "s_op", "kwargs": {"method": "specific"}}],
    }
    registry.register_preset("test_preset", preset_cfg)

    color_ops = registry.get_operations_for_preset("test_preset", category="color")
    assert isinstance(color_ops, list) and len(color_ops) == 1
    assert isinstance(color_ops[0], DummyColorOp)
    assert color_ops[0].params["color"] == 3

    select_ops = registry.get_operations("select", "test_preset") # Using the alias method
    assert isinstance(select_ops, list) and len(select_ops) == 1
    assert isinstance(select_ops[0], DummySelectOp)
    assert select_ops[0].params["method"] == "specific"

def test_get_operations_for_preset_all_categories(registry: OperationRegistry):
    registry.register_operation("color", "c_op", DummyColorOp)
    registry.register_operation("select", "s_op", DummySelectOp)
    preset_cfg = {
        "color": [{"name": "c_op", "kwargs": {"color": 7}}],
        "select": [{"name": "s_op", "kwargs": {"method": "all_cat_test"}}],
    }
    registry.register_preset("all_cat_preset", preset_cfg)

    all_ops = registry.get_operations_for_preset("all_cat_preset") # Category=None
    assert isinstance(all_ops, dict)
    assert "color" in all_ops and "select" in all_ops
    assert len(all_ops["color"]) == 1 and isinstance(all_ops["color"][0], DummyColorOp)
    assert all_ops["color"][0].params["color"] == 7
    assert len(all_ops["select"]) == 1 and isinstance(all_ops["select"][0], DummySelectOp)
    assert all_ops["select"][0].params["method"] == "all_cat_test"

def test_get_operations_preset_not_found(registry: OperationRegistry):
    with pytest.raises(OperationError, match="not found"):
        registry.get_operations_for_preset("ghost_preset")

def test_get_operations_category_not_in_preset(registry: OperationRegistry):
    registry.register_operation("color", "c_op", DummyColorOp)
    preset_cfg = {"color": [{"name": "c_op", "kwargs": {}}]}
    registry.register_preset("cat_test_preset", preset_cfg)

    # Requesting a category that exists in registry but not in this preset's config
    with pytest.raises(OperationError, match="not found in preset"):
        registry.get_operations_for_preset("cat_test_preset", category="non_preset_category")

    # Using the alias method get_operations should also fail or return empty
    # The current get_operations_for_preset raises if category is specified but not in preset.
    # If it returned empty list, this would pass:
    # assert registry.get_operations("non_preset_category", "cat_test_preset") == []


def test_get_operations_instantiation_failure(registry: OperationRegistry):
    # Register op class correctly
    registry.register_operation("test", "op_good_class", DummyColorOp)
    # Preset config refers to it, but if DummyColorOp's __init__ was changed to require
    # a param not provided in kwargs, or if kwargs type is wrong, it would fail.
    # Example: DummyColorOp expects 'color' int. If kwargs give string, it might fail in op's init.
    # For this test, let's simulate an op whose class is registered but cannot be found during preset instantiation
    # (e.g., if registry._operations was manually corrupted after registration but before preset use)

    # This is hard to test without breaking registry internals or having an op that fails on init.
    # Let's assume an op in preset refers to a name not in its category in _operations (inconsistency).
    # This should be caught by register_preset, but if registry was modified:

    # Correct registration:
    registry.register_operation("category_A", "op_A1", DummyColorOp)
    preset_conf = {"category_A": [{"name": "op_A1", "kwargs": {"color": 1}}]}
    registry.register_preset("init_fail_test_preset", preset_conf)

    # Corrupt registry to make op_A1 (for category_A) unfindable by get_operation_class
    # This simulates an internal inconsistency.
    original_op_class_store = registry._operations["category_A"]["op_A1"]
    del registry._operations["category_A"]["op_A1"]

    with pytest.raises(OperationError, match="class for 'op_A1' .* not found in registry"):
        registry.get_operations_for_preset("init_fail_test_preset", category="category_A")

    # Restore for other tests if registry is module-scoped (it's function-scoped here)
    registry._operations["category_A"]["op_A1"] = original_op_class_store


# --- Tests for Listing ---
def test_list_available_presets(registry: OperationRegistry):
    assert registry.list_available_presets() == []
    registry.register_operation("color", "c", DummyColorOp) # Need op for valid preset
    registry.register_preset("p1", {"color": [{"name": "c", "kwargs": {}}]})
    registry.register_preset("p2", {"color": [{"name": "c", "kwargs": {}}]})
    assert sorted(registry.list_available_presets()) == ["p1", "p2"]

def test_list_available_operations(registry: OperationRegistry):
    assert registry.list_available_operations() == {} # No categories, no ops
    registry.register_operation("cat1", "op1a", DummyColorOp)
    registry.register_operation("cat1", "op1b", DummySelectOp)
    registry.register_operation("cat2", "op2a", DummyColorOp)

    all_ops = registry.list_available_operations() # type: ignore # No arg = dict
    assert isinstance(all_ops, dict)
    assert sorted(all_ops.keys()) == ["cat1", "cat2"]
    assert sorted(all_ops["cat1"]) == ["op1a", "op1b"]
    assert sorted(all_ops["cat2"]) == ["op2a"]

    cat1_ops = registry.list_available_operations(category="cat1") # type: ignore # With arg = list
    assert isinstance(cat1_ops, list)
    assert sorted(cat1_ops) == ["op1a", "op1b"]

    assert registry.list_available_operations(category="non_existent_cat") == []


# Add more tests:
# - Edge cases for preset configurations (empty kwargs, empty op lists for a category).
# - Robustness of get_operations if an op's __init__ raises an error. (Partially covered by init_fail_test_preset)
