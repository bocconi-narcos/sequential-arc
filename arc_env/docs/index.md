# ARC Environment Documentation

Welcome to the official documentation for `arc_env`, a Python package providing a [Gymnasium](https://gymnasium.farama.org/) environment for the Abstraction and Reasoning Corpus (ARC).

`arc_env` is designed to be a comprehensive, extensible, and user-friendly platform for researchers and developers working on ARC challenge solvers, particularly those leveraging reinforcement learning, search algorithms, or program synthesis.

## Key Features

*   **Gymnasium Compliant**: Standard `gym.Env` interface for easy integration with RL libraries and algorithms.
*   **Modular Design**: Clear separation of concerns for components like data loading, action spaces, observation spaces, DSL operations, and solvers.
*   **Extensible DSL**: A Domain-Specific Language (DSL) for defining actions and transformations on ARC grids, with a registry for custom operations and presets.
*   **Configurable**: Flexible configuration system for environment parameters, action spaces, and solvers.
*   **Data Management**: Robust data loaders for ARC tasks, supporting standard formats and custom sources.
*   **Wrappers**: Environment wrappers for common needs like observation normalization, action masking, and curriculum learning.
*   **Solver System**: A base structure for implementing and evaluating different ARC solvers.
*   **Comprehensive Testing**: Includes unit and integration tests (to be populated).
*   **Examples and Documentation**: Usage examples and this documentation site to help you get started.

## Table of Contents

*   **Getting Started**
    *   [Installation](./tutorials/getting_started.md#installation) (Link to be updated if structure changes)
    *   [Basic Usage](./tutorials/basic_usage_link.md) (Link to example script or tutorial page)
*   **Tutorials**
    *   [Overview](./tutorials/index.md)
    *   [Creating Custom Solvers](./tutorials/custom_solver_link.md)
    *   [Working with the DSL](./tutorials/dsl_tutorial_link.md)
*   **API Reference**
    *   [Overview](./api/index.md)
    *   `arc_env.environments`
    *   `arc_env.spaces`
    *   `arc_env.dsl`
    *   `arc_env.solvers`
    *   (More modules...)
*   **Design Documents**
    *   [Overview](./design/index.md)
    *   [Package Architecture](./design/architecture.md)
*   **Contributing**
    *   (Link to CONTRIBUTING.md if it exists)
*   **Changelog**
    *   [View Changelog](./changelog.md)

## Citing `arc_env`

If you use `arc_env` in your research, please consider citing it (details TBD once published or formally released).

---

*This documentation is currently under development.*
