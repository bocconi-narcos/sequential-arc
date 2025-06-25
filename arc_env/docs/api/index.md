# API Reference

This section provides detailed API documentation for the `arc_env` package.
The documentation is generated from the source code docstrings (e.g., using Sphinx with autodoc).

## Modules

Below is a list of the core modules within `arc_env`. Click on a module to see its classes, functions, and methods.

*   **`arc_env.arc_env.environments`**:
    *   [`arc_env`](./environments/arc_env.md): Main `ARCEnv` class.
    *   [`base_env`](../core/base_env.md): Abstract `BaseARCEnvironment`. (Link might need adjustment based on final doc structure)
    *   `wrappers`:
        *   [`base`](./environments/wrappers/base.md): `BaseARCWrapper`.
        *   [`action_masking`](./environments/wrappers/action_masking.md)
        *   [`curriculum`](./environments/wrappers/curriculum.md)
        *   [`normalization`](./environments/wrappers/normalization.md)
    *   `variants`:
        *   [`mini_arc`](./environments/variants/mini_arc.md)
        *   [`multi_task`](./environments/variants/multi_task.md)

*   **`arc_env.arc_env.spaces`**:
    *   [`action_spaces`](./spaces/action_spaces.md): `ARCActionSpace`.
    *   [`observation_spaces`](./spaces/observation_spaces.md): `ARCObservationSpace`.
    *   [`base_action_space`](../core/base_action_space.md) (Link adjustment)
    *   [`base_observation_space`](../core/base_observation_space.md) (Link adjustment)
    *   [`space_utils`](./spaces/space_utils.md)

*   **`arc_env.arc_env.dsl`**: Domain-Specific Language components.
    *   `core`:
        *   [`base_operations`](./dsl/core/base_operations.md)
        *   [`operation_registry`](./dsl/core/operation_registry.md)
    *   `operations`: Concrete DSL operations (color, selection, etc.)
        *   [`color`](./dsl/operations/color.md)
        *   *(More operation categories)*
    *   `presets`: DSL operation presets.
        *   [`default`](./dsl/presets/default.md)
        *   *(More presets)*
    *   `utils`: DSL utility functions.
        *   [`grid_utils`](./dsl/utils/grid_utils.md)
        *   *(More DSL utils)*

*   **`arc_env.arc_env.config`**: Configuration classes.
    *   [`base`](./config/base.md): `BaseConfig`.
    *   [`environment`](./config/environment.md): `EnvironmentConfig`.
    *   *(More config types)*

*   **`arc_env.arc_env.data`**: Data loading and processing.
    *   `loaders`:
        *   [`base`](./data/loaders/base.md)
        *   [`arc_loader`](./data/loaders/arc_loader.md)
    *   `processors`:
        *   [`validation`](./data/processors/validation.md)
    *   `datasets`:
        *   [`arc_dataset`](./data/datasets/arc_dataset.md)

*   **`arc_env.arc_env.solvers`**: Solver system.
    *   `base`:
        *   [`base_solver`](./solvers/base/base_solver.md)
        *   [`solver_registry`](./solvers/base/solver_registry.md)
    *   `implementations`: Placeholder concrete solvers.
    *   `evaluation`:
        *   [`metrics`](./solvers/evaluation/metrics.md)
        *   [`benchmarking`](./solvers/evaluation/benchmarking.md)

*   **`arc_env.arc_env.utils`**: General utility modules.
    *   [`logging`](./utils/logging.md)
    *   [`serialization`](./utils/serialization.md)

*   **`arc_env.arc_env.exceptions`**: Custom exceptions.
    *   [`exceptions`](./exceptions.md)

---
*This API reference will be auto-generated. Links and content are placeholders.*
