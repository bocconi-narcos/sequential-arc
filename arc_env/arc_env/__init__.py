# arc_env.arc_env (main library module) __init__.py

# This file defines the public API of the `arc_env.arc_env` module itself.
# Many of these will also be exposed by the top-level `arc_env/__init__.py`.

# Version
from .version import __version__

# Core Abstractions and Protocols
from .core.base_env import BaseARCEnvironment
from .core.base_action_space import BaseActionSpace
from .core.base_observation_space import BaseObservationSpace
from .core.protocols import (
    EnvironmentProtocol, ActionSpaceProtocol, ObservationSpaceProtocol,
    SolverProtocol, DataLoaderProtocol, DataProcessorProtocol, OperationProtocol
)

# Main Environment Classes
from .environments.arc_env import ARCEnv
from .environments.variants.mini_arc import MiniARCEnv
from .environments.variants.multi_task import MultiTaskARCEnv

# Environment Wrappers (expose base and perhaps most common ones)
from .environments.wrappers.base import BaseARCWrapper
from .environments.wrappers.action_masking import ActionMaskingWrapper
from .environments.wrappers.curriculum import CurriculumWrapper
from .environments.wrappers.normalization import ObservationNormalizationWrapper

# Configuration Classes
from .config.base import BaseConfig
from .config.environment import EnvironmentConfig
from .config.action_space import ActionSpaceConfig
from .config.solver import SolverConfig, HeuristicSolverConfig, RLSolverConfig
from .config.validation import validate_master_config, ConfigValidationError

# Spaces
from .spaces.action_spaces import ARCActionSpace, DecodedArcOps
from .spaces.observation_spaces import ARCObservationSpace, ARCStructuredObservation
from .spaces.space_utils import flatten_space, unflatten_action, get_space_size, create_empty_sample

# DSL (Domain-Specific Language) Core Components
from .dsl.core.base_operations import BaseOperation
from .dsl.core.operation_registry import OperationRegistry
# Specific operations, presets, or DSL utils can be exposed if deemed part of primary API
# e.g., from .dsl.operations.color import FillOperation
# e.g., from .dsl.presets.default import DefaultPresetLoader

# Data Management
from .data.loaders.base import BaseDataLoader, ARCTaskData # Protocol
from .data.loaders.arc_loader import ARCFileLoader, ARCCombinedFileLoader
from .data.loaders.custom_loader import CustomDataLoader
from .data.processors.base import BaseDataProcessor
from .data.processors.validation import ARCTaskValidator
from .data.processors.augmentation import GridAugmenter, ARCTaskAugmenter, random_flip, random_rotate90
from .data.datasets.arc_dataset import ARCDataset, ARCDatasetItem
from .data.datasets.challenge_dataset import ChallengeDataset, ChallengeDatasetItem

# Solvers
from .solvers.base.base_solver import BaseSolver
from .solvers.base.solver_registry import SolverRegistry
from .solvers.base.solver_utils import compare_grids, solution_accuracy
# Placeholder solvers are typically not part of the main API surface unless for direct use/example
# from .solvers.implementations.heuristic.placeholder_heuristic_solver import PlaceholderHeuristicSolver
from .solvers.evaluation.metrics import TaskEvaluationResult, evaluate_single_task_attempt
from .solvers.evaluation.benchmarking import BenchmarkRunner

# Utilities (selectively)
from .utils.logging import setup_logger, get_logger
from .utils.random import set_global_seed, get_numpy_rng
from .utils.serialization import save_to_json, load_from_json, save_to_pickle, load_from_pickle, NpEncoder
from .utils.performance import time_function, CodeTimer, CallProfiler

# Custom Exceptions
from .exceptions import * # Import all custom exceptions

# Define __all__ for `from arc_env.arc_env import *`
# This should be comprehensive for the `arc_env.arc_env` module's public interface.
__all__ = [
    "__version__",
    # Core Abstractions & Protocols
    "BaseARCEnvironment", "BaseActionSpace", "BaseObservationSpace",
    "EnvironmentProtocol", "ActionSpaceProtocol", "ObservationSpaceProtocol",
    "SolverProtocol", "DataLoaderProtocol", "DataProcessorProtocol", "OperationProtocol",
    # Environments
    "ARCEnv", "MiniARCEnv", "MultiTaskARCEnv",
    # Wrappers
    "BaseARCWrapper", "ActionMaskingWrapper", "CurriculumWrapper", "ObservationNormalizationWrapper",
    # Configs
    "BaseConfig", "EnvironmentConfig", "ActionSpaceConfig", "SolverConfig",
    "HeuristicSolverConfig", "RLSolverConfig", "validate_master_config", "ConfigValidationError",
    # Spaces
    "ARCActionSpace", "DecodedArcOps", "ARCObservationSpace", "ARCStructuredObservation",
    "flatten_space", "unflatten_action", "get_space_size", "create_empty_sample",
    # DSL
    "BaseOperation", "OperationRegistry",
    # Data
    "BaseDataLoader", "ARCTaskData", "ARCFileLoader", "ARCCombinedFileLoader", "CustomDataLoader",
    "BaseDataProcessor", "ARCTaskValidator", "GridAugmenter", "ARCTaskAugmenter", "random_flip", "random_rotate90",
    "ARCDataset", "ARCDatasetItem", "ChallengeDataset", "ChallengeDatasetItem",
    # Solvers
    "BaseSolver", "SolverRegistry", "compare_grids", "solution_accuracy",
    "TaskEvaluationResult", "evaluate_single_task_attempt", "BenchmarkRunner",
    # Utilities
    "setup_logger", "get_logger", "set_global_seed", "get_numpy_rng",
    "save_to_json", "load_from_json", "save_to_pickle", "load_from_pickle", "NpEncoder",
    "time_function", "CodeTimer", "CallProfiler",
    # Exceptions (imported via *)
    "ARCError", "ConfigurationError", "DataLoadError", "DataProcessingError", "OperationError",
    "DSLExecutionError", "TaskNotSetError", "InvalidActionError", "SolverError",
    "ModelNotFoundError", "SolverPredictionError",
]
