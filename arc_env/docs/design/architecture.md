# `arc_env` Package Architecture

This document provides a high-level overview of the architecture of the `arc_env` Python package. It describes the main components, their responsibilities, and how they interact.

## Core Principles

The architecture is guided by the following principles:

*   **Modularity**: Components are designed to be as independent as possible, with clear interfaces.
*   **Extensibility**: Users should be able to easily add new environments, actions, DSL operations, data loaders, and solvers.
*   **Gymnasium Compliance**: The core environment adheres to the `gymnasium.Env` interface for broad compatibility.
*   **Configuration**: Key aspects of the environment and its components are configurable via a structured configuration system.
*   **Clarity**: Code and structure aim for readability and ease of understanding.

## Main Components

The package is organized into several main modules/directories within `arc_env/arc_env/`:

1.  **`core/`**:
    *   **Purpose**: Defines the fundamental abstract base classes (ABCs) and protocols that form the backbone of the package.
    *   **Key Files**:
        *   `base_env.py`: `BaseARCEnvironment` (ABC for all ARC environments).
        *   `base_action_space.py`: `BaseActionSpace` (ABC for custom action spaces).
        *   `base_observation_space.py`: `BaseObservationSpace` (ABC for custom observation spaces).
        *   `protocols.py`: `typing.Protocol` definitions for structural subtyping (e.g., `EnvironmentProtocol`, `SolverProtocol`).
    *   **Interaction**: Other components (environments, spaces, solvers) inherit from or implement these core abstractions/protocols.

2.  **`environments/`**:
    *   **Purpose**: Contains concrete implementations of ARC environments.
    *   **Key Files**:
        *   `arc_env.py`: The main `ARCEnv` class, which integrates most other components.
        *   `wrappers/`: Sub-package for environment wrappers (e.g., normalization, action masking, curriculum).
        *   `variants/`: Sub-package for environment variants (e.g., `MiniARCEnv`, `MultiTaskARCEnv`).
    *   **Interaction**: `ARCEnv` uses `config`, `spaces`, `data`, and potentially `dsl` components. Wrappers modify the behavior of an `ARCEnv` instance.

3.  **`spaces/`**:
    *   **Purpose**: Defines the action and observation spaces specific to ARC.
    *   **Key Files**:
        *   `action_spaces.py`: `ARCActionSpace` (handles DSL operations, factorized/joint modes).
        *   `observation_spaces.py`: `ARCObservationSpace` (defines the structure of observations like task grid, train pairs).
        *   `space_utils.py`: Utility functions related to Gymnasium spaces.
    *   **Interaction**: Used by `ARCEnv` to define `env.action_space` and `env.observation_space`. `ARCActionSpace` interacts with the `dsl.core.OperationRegistry`.

4.  **`dsl/`** (Domain-Specific Language):
    *   **Purpose**: Implements the system for defining and managing grid transformation operations.
    *   **Key Files**:
        *   `core/base_operations.py`: `BaseOperation` ABC.
        *   `core/operation_registry.py`: `OperationRegistry` for DSL ops and presets.
        *   `operations/`: Concrete operation implementations (e.g., `color.py`, `selection.py`).
        *   `presets/`: Definitions for sets of operations (e.g., `default.py`, `minimal.py`).
        *   `utils/`: Utilities specific to DSL grid manipulations (e.g., `grid_utils.py`).
    *   **Interaction**: `ARCActionSpace` uses `OperationRegistry` to populate its available actions. `ARCEnv` might use DSL operations internally to apply actions.

5.  **`data/`**:
    *   **Purpose**: Handles loading, processing, and managing ARC task data.
    *   **Key Files**:
        *   `loaders/`: Data loader implementations (`ARCFileLoader`, `ARCCombinedFileLoader`, `CustomDataLoader`, `BaseDataLoader` ABC).
        *   `processors/`: Data processors (`ARCTaskValidator`, `ARCTaskAugmenter`, `BaseDataProcessor` ABC).
        *   `datasets/`: PyTorch-style `Dataset` classes (`ARCDataset`, `ChallengeDataset`).
        *   `assets/`: Placeholder for static data files (e.g., default `challenges.json`).
    *   **Interaction**: `ARCEnv` uses a data loader. `ARCDataset` uses loaders and processors.

6.  **`solvers/`**:
    *   **Purpose**: Provides a framework for implementing and evaluating ARC task solvers.
    *   **Key Files**:
        *   `base/`: `BaseSolver` ABC, `SolverRegistry`, `solver_utils.py`.
        *   `implementations/`: Placeholder concrete solvers (heuristic, RL, hybrid).
        *   `evaluation/`: `metrics.py`, `benchmarking.py` (`BenchmarkRunner`).
    *   **Interaction**: Solvers interact with `ARCEnv` (or its observation/action spaces). `BenchmarkRunner` uses `ARCDataset`, `BaseSolver`, and `ARCEnv`.

7.  **`config/`**:
    *   **Purpose**: Defines configuration objects for various components.
    *   **Key Files**: `base.py` (`BaseConfig`), `environment.py` (`EnvironmentConfig`), `action_space.py`, `solver.py`.
    *   **Interaction**: Configuration objects are passed to the constructors of main components like `ARCEnv`, `ARCActionSpace`, solvers, etc.

8.  **`utils/`**:
    *   **Purpose**: General utility modules not specific to one component.
    *   **Key Files**: `logging.py`, `random.py`, `serialization.py`, `performance.py`.
    *   **Interaction**: Used by various parts of the package.

9.  **`exceptions.py`**:
    *   **Purpose**: Defines custom exception classes for the package.
    *   **Interaction**: Raised and caught throughout the codebase for specific error conditions.

## Top-Level Structure

The project also includes top-level files and directories:

*   `arc_env/` (root directory for the installable package)
    *   `__init__.py`: Makes `arc_env` a package, might export key classes/functions.
    *   `arc_env/`: The main source code directory described above.
    *   `py.typed`: PEP 561 marker file for type checking.
    *   `version.py`: Single source of truth for package version (within `arc_env/arc_env/`).
*   `tests/`: Unit and integration tests.
*   `examples/`: Usage example scripts.
*   `docs/`: This documentation.
*   `scripts/`: Utility scripts for development or data management.
*   `requirements/`: Requirement files (`base.txt`, `dev.txt`, etc.).
*   `pyproject.toml`: Modern Python packaging configuration.
*   `setup.py`: Legacy packaging support, build script.
*   `README.md`, `LICENSE`, `CONTRIBUTING.md`, etc.

## Data Flow (Conceptual Example: Agent Interaction)

1.  **Initialization**:
    *   User creates `EnvironmentConfig`, `ActionSpaceConfig`, etc.
    *   User creates an `ARCEnv` instance, passing configs and a `BaseDataLoader`.
    *   `ARCEnv` initializes its `ARCObservationSpace` and `ARCActionSpace` (which uses `OperationRegistry` populated by preset loaders).
2.  **Episode Start**:
    *   User (or training loop) calls `env.reset()`.
    *   `ARCEnv` loads a task using its data loader (e.g., based on `initial_task_id` or curriculum logic if wrapped).
    *   Internal state (`current_grid`, `target_grid`) is initialized.
    *   An initial `ARCStructuredObservation` is returned.
3.  **Agent Takes Action**:
    *   Agent (or `BaseSolver`) receives observation, predicts an action compatible with `env.action_space`.
    *   User calls `env.step(action)`.
4.  **Environment Processes Action**:
    *   `ARCEnv` calls `action_space.decode(action)` to get `DecodedArcOps`.
    *   `ARCEnv` applies these operations to its `current_grid` (potentially using DSL operations).
    *   Reward is calculated, `terminated`/`truncated` flags are determined.
    *   New observation and info dict are prepared.
5.  **Cycle**: Step 3-4 repeat until `terminated` or `truncated`. Then back to Step 2.

This architecture aims to provide a robust and flexible foundation for working with the ARC challenge.
