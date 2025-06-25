# Tutorials

This section provides tutorials to help you get started with `arc_env` and understand its various components and functionalities.

## Available Tutorials

1.  **[Getting Started with `arc_env`](./getting_started.md)**
    *   Covers installation, basic environment setup, and running a simple interaction loop. A good starting point for all users.

2.  **[Understanding Actions and Observations](./actions_observations.md)** (Placeholder)
    *   Detailed explanation of the `ARCActionSpace` (factorized vs. joint modes) and `ARCObservationSpace` structure.
    *   How to interpret observations and construct valid actions.

3.  **[Working with the Domain-Specific Language (DSL)](./dsl_overview.md)** (Placeholder)
    *   Introduction to the DSL operations, presets, and the operation registry.
    *   How to define custom operations or presets.
    *   Using the DSL to interact with the environment programmatically.

4.  **[Loading and Managing ARC Task Data](./data_management.md)** (Placeholder)
    *   Using `ARCFileLoader`, `ARCCombinedFileLoader`, and `CustomDataLoader`.
    *   Working with `ARCDataset` and `ChallengeDataset`.
    *   Data validation and augmentation basics.

5.  **[Implementing Custom Solvers](./custom_solvers.md)** (Placeholder)
    *   Step-by-step guide to creating your own solver by inheriting from `BaseSolver`.
    *   Registering your solver with the `SolverRegistry`.
    *   Integrating your solver with the environment and evaluation tools.

6.  **[Using Environment Wrappers](./environment_wrappers.md)** (Placeholder)
    *   Overview of available wrappers (`ObservationNormalizationWrapper`, `ActionMaskingWrapper`, `CurriculumWrapper`).
    *   How to apply wrappers to modify environment behavior.
    *   Creating custom wrappers.

7.  **[Evaluating Solvers with BenchmarkRunner](./evaluating_solvers.md)** (Placeholder)
    *   Setting up and running benchmarks using `BenchmarkRunner`.
    *   Understanding the output metrics and evaluation results.

8.  **[Training an RL Agent (Conceptual)](./training_rl_agent.md)** (Placeholder)
    *   High-level guide on how `arc_env` can be integrated with common RL frameworks.
    *   Considerations for observation/action spaces, reward shaping, and multi-task learning.

## Prerequisites

Most tutorials assume you have:
*   Successfully installed `arc_env` (see [Getting Started](./getting_started.md)).
*   Basic familiarity with Python and NumPy.
*   Understanding of the ARC challenge itself.
*   For RL-related tutorials, some background in reinforcement learning concepts is helpful.

---
*More tutorials will be added. If you have suggestions for tutorial topics, please let us know!*
