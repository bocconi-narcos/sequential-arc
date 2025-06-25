# Changelog

All notable changes to the `arc_env` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (hopefully!).

## [Unreleased] - YYYY-MM-DD

### Added
- Initial project structure setup.
- Core abstractions: `BaseARCEnvironment`, `BaseActionSpace`, `BaseObservationSpace`, `Protocols`.
- Configuration system: `BaseConfig`, `EnvironmentConfig`, `ActionSpaceConfig`, `SolverConfig`, validation utilities.
- Domain-Specific Language (DSL) system:
    - Core: `BaseOperation`, `OperationRegistry`.
    - Placeholder operations: `color`, `selection`, `transformation`, `composition`.
    - Preset loaders: `DefaultPresetLoader`, `MinimalPresetLoader`, `CustomPresetLoader`.
    - DSL utils: `grid_utils`, `color_utils`, `background_utils`.
- Spaces: `ARCActionSpace`, `ARCObservationSpace`, `space_utils`.
- Data management:
    - Loaders: `BaseDataLoader`, `ARCFileLoader`, `ARCCombinedFileLoader`, `CustomDataLoader` (placeholder).
    - Processors: `BaseDataProcessor`, `ARCTaskValidator`, `ARCTaskAugmenter`.
    - Datasets: `ARCDataset`, `ChallengeDataset`.
    - Placeholder asset files: `challenges.json`, `solutions.json`.
- Environment Wrappers: `BaseARCWrapper`, `ObservationNormalizationWrapper`, `ActionMaskingWrapper`, `CurriculumWrapper`.
- Main Environment: `ARCEnv` implementation and variants (`MiniARCEnv`, `MultiTaskARCEnv` placeholders).
- Solver System:
    - Base: `BaseSolver`, `SolverRegistry`, `solver_utils`.
    - Implementations: Placeholders for heuristic, RL, and hybrid solvers.
    - Evaluation: `metrics.py`, `benchmarking.py` (`BenchmarkRunner`).
- General Utilities: `logging`, `random`, `serialization`, `performance` modules.
- Custom Exceptions: Defined in `arc_env.exceptions`.
- Test Structure: `conftest.py`, placeholder unit and integration tests, fixture files.
- Example Scripts: `basic_usage.py`, `custom_solver.py`, `training_example.py` (conceptual), `benchmarking.py`.
- Documentation Structure: Initial `docs/` layout with placeholder markdown files.
- Packaging: `pyproject.toml`, `setup.py`.
- Type checking: `py.typed` marker.
- Versioning: `arc_env/arc_env/version.py`.

### Changed
- (Nothing changed from a previous version yet)

### Deprecated
- (Nothing deprecated yet)

### Removed
- (Nothing removed yet)

### Fixed
- (No bugs fixed yet as it's initial setup)

### Security
- (No security vulnerabilities addressed yet)

## [0.1.0] - YYYY-MM-DD (Planned first release)
- (Details of the first release will go here)

---
*This changelog will be updated as new versions are released or significant changes are made.*
