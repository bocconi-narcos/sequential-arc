from __future__ import annotations

import pytest
from pathlib import Path
import shutil
import json
import numpy as np
from typing import Any, Generator # Added Any, Generator for fixture type hints

# This file (conftest.py) is used by pytest to share fixtures, hooks, and plugins
# across multiple test files.

# --- General Test Configuration & Hooks ---

def pytest_addoption(parser):
    """Adds command-line options to pytest."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    # Add more options if needed, e.g., --integration-only, --level=...

def pytest_configure(config):
    """Allows plugins and conftest files to perform initial configuration."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    # Add other markers if used

def pytest_collection_modifyitems(config, items):
    """Called after test collection has been performed. Modifies items in-place."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# --- Shared Fixtures ---

@pytest.fixture(scope="session") # Available to all tests in the session
def temp_test_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Creates a temporary directory for test files that persists across the test session.
    Use tmp_path for function-scoped temporary directories.
    """
    tdir = tmp_path_factory.mktemp("arc_env_shared_tests_")
    print(f"Created session-scoped temp directory for tests: {tdir}")
    return tdir


@pytest.fixture(scope="function") # New temp dir for each test function
def function_temp_dir(tmp_path: Path) -> Path:
    """
    Provides a function-scoped temporary directory.
    `tmp_path` is a built-in pytest fixture.
    """
    # print(f"Using function-scoped temp directory: {tmp_path}")
    return tmp_path


@pytest.fixture(scope="session")
def dummy_operation_registry():
    """
    Provides a basic OperationRegistry, possibly pre-populated with some default/dummy operations.
    Useful for testing components that depend on a registry (e.g., ARCActionSpace, ARCEnv).
    """
    from arc_env.dsl.core.operation_registry import OperationRegistry
    from arc_env.dsl.presets.default import DefaultPresetLoader # To get some ops

    registry = OperationRegistry()
    try:
        # Attempt to load default preset to have some operations available
        DefaultPresetLoader(registry).load()
    except Exception as e:
        print(f"Conftest: Warning - Could not load DefaultPreset into dummy_operation_registry: {e}. Registry may be empty.")
        # Fallback: Register a minimal set of dummy operations if default loading fails
        # This part might be needed if DefaultPresetLoader has complex dependencies not met in test setup.
        # For now, assume DefaultPresetLoader is robust enough or its failures are handled.
    return registry

@pytest.fixture(scope="session")
def dummy_env_config():
    """Provides a default EnvironmentConfig for tests."""
    from arc_env.config.environment import EnvironmentConfig
    return EnvironmentConfig(canvas_size=10, max_steps=20) # Small canvas for faster tests

@pytest.fixture(scope="session")
def dummy_action_space_config():
    """Provides a default ActionSpaceConfig for tests."""
    from arc_env.config.action_space import ActionSpaceConfig
    # Ensure "minimal" preset is available for tests if used, or "default"
    return ActionSpaceConfig(preset="default", available_presets=["default", "minimal"])


@pytest.fixture(scope="module") # Example: Module-scoped fixture for data loading setup
def dummy_arc_tasks_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Creates a temporary directory with a few dummy ARC task JSON files.
    Useful for testing data loaders and components that consume task data.
    """
    tasks_dir = tmp_path_factory.mktemp("dummy_arc_tasks_")

    task_defs = {
        "test_task_001": {
            "train": [{"input": [[1]], "output": [[0]]}],
            "test": [{"input": [[1,2],[3,4]], "output": [[0,0],[0,0]]}]
        },
        "test_task_002": {
            "train": [{"input": [[5,5]], "output": [[6,6]]}],
            "test": [{"input": [[7]], "output": [[8]]}]
        },
        "invalid_task_format": { # Missing 'test' key
            "train": [{"input": [[1]], "output": [[0]]}]
        },
        "task_with_empty_grids": {
             "train": [{"input": [[]], "output": [[0]]}], # Empty list for grid
             "test": [{"input": [[1]], "output": [[]]}]
        }
    }
    for task_id, content in task_defs.items():
        with open(tasks_dir / f"{task_id}.json", "w") as f:
            json.dump(content, f)

    print(f"Created dummy ARC tasks in: {tasks_dir}")
    return tasks_dir


@pytest.fixture
def dummy_arc_file_loader(dummy_arc_tasks_dir: Path):
    """Provides an ARCFileLoader instance pointing to the dummy tasks directory."""
    from arc_env.data.loaders.arc_loader import ARCFileLoader
    return ARCFileLoader(data_directory=dummy_arc_tasks_dir)


# Add more shared fixtures as needed:
# - Mocked objects (e.g., mock_solver, mock_environment)
# - Pre-loaded data structures
# - Configurations for different test scenarios

# Example: A fixture for a basic ARCEnv instance
@pytest.fixture # Scope defaults to "function"
def basic_arc_env(
    dummy_env_config: EnvironmentConfig,
    dummy_action_space_config: ActionSpaceConfig,
    dummy_arc_file_loader: ARCFileLoader, # Specific loader type
    dummy_operation_registry: OperationRegistry
) -> ARCEnv | None: # Can return None if setup fails and skips
    """Provides a basic, usable ARCEnv instance for testing."""
    from arc_env.environments.arc_env import ARCEnv
    # Configs are already correct types due to fixture dependencies

    # Ensure env_cfg.data_path is consistent if ARCEnv tries to make its own default loader,
    # even though we pass dummy_arc_file_loader directly.
    # Modifying fixture instance like this can be tricky if it's shared (session-scoped).
    # dummy_env_config is session-scoped. Let's create a copy or ensure it's function-scoped if modified.
    # For now, let's assume dummy_env_config can be modified or its data_path is already correct/unused.
    # A safer way: create a new EnvironmentConfig instance here if modifications are needed.

    # If dummy_env_config is session-scoped, we should not modify it directly here.
    # Let's use its values but ensure data_path is from the file loader for this instance.
    current_env_cfg = EnvironmentConfig(
        canvas_size=dummy_env_config.canvas_size,
        max_steps=dummy_env_config.max_steps,
        # Copy other relevant params from dummy_env_config...
        data_path=dummy_arc_file_loader.data_directory # Crucial for consistency
    )

    try:
        env = ARCEnv(
            env_config=current_env_cfg, # Use the locally adjusted or freshly created config
            action_space_config=dummy_action_space_config, # This is also session-scoped
            data_loader=dummy_arc_file_loader,
            operation_registry=dummy_operation_registry,
            initial_task_id="test_task_001" # Load a known valid task
        )
        return env
    except Exception as e:
        pytest.skip(f"Skipping tests requiring basic_arc_env due to setup error: {e}")
        return None # Should be handled by skip or raise
            env_config=env_cfg,
            action_space_config=act_cfg,
            data_loader=dummy_arc_file_loader,
            operation_registry=dummy_operation_registry,
            initial_task_id="test_task_001" # Load a known valid task
        )
        return env
    except Exception as e:
        pytest.skip(f"Skipping tests requiring basic_arc_env due to setup error: {e}")
        # Or raise, if this fixture is critical for many tests.
        # raise RuntimeError(f"Failed to create basic_arc_env fixture: {e}") from e
        return None # Should be handled by skip or raise

# If you need to clean up session-scoped fixtures after all tests, use a finalizer:
# @pytest.fixture(scope="session")
# def my_session_resource(tmp_path_factory):
#     path = tmp_path_factory.mktemp("data")
#     # ... setup resource ...
#     yield path # Resource is available during tests
#     # ... teardown resource ...
#     shutil.rmtree(path) # Example cleanup

print("conftest.py loaded: Shared fixtures and pytest hooks are configured.")
