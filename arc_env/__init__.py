# arc_env (distribution package) top-level __init__.py

# Expose the version of the package
from .arc_env.version import __version__

# Expose key components from the main library module (arc_env.arc_env)
# This makes them accessible like: `from arc_env import ARCEnv`
# instead of `from arc_env.arc_env import ARCEnv` (if arc_env.arc_env is the library root)

# Core Environment
from .arc_env.environments.arc_env import ARCEnv
from .arc_env.environments.variants.mini_arc import MiniARCEnv
from .arc_env.environments.variants.multi_task import MultiTaskARCEnv

# Core Abstractions (optional to expose all, but BaseARCEnvironment is key)
from .arc_env.core.base_env import BaseARCEnvironment
from .arc_env.core.base_action_space import BaseActionSpace
from .arc_env.core.base_observation_space import BaseObservationSpace

# Main Spaces (if users need to import them directly often)
from .arc_env.spaces.action_spaces import ARCActionSpace
from .arc_env.spaces.observation_spaces import ARCObservationSpace, ARCStructuredObservation # TypedDict

# Config classes (users will likely need these)
from .arc_env.config.base import BaseConfig
from .arc_env.config.environment import EnvironmentConfig
from .arc_env.config.action_space import ActionSpaceConfig
from .arc_env.config.solver import SolverConfig, HeuristicSolverConfig, RLSolverConfig

# DSL Core (Registry is important if users want to customize DSL ops)
from .arc_env.dsl.core.operation_registry import OperationRegistry
from .arc_env.dsl.core.base_operations import BaseOperation

# Data Loading and Datasets (if these are part of the public API for users)
from .arc_env.data.loaders.base import BaseDataLoader # ABC/Protocol
from .arc_env.data.loaders.arc_loader import ARCFileLoader, ARCCombinedFileLoader
from .arc_env.data.datasets.arc_dataset import ARCDataset
from .arc_env.data.datasets.challenge_dataset import ChallengeDataset

# Solvers (Base and Registry are important)
from .arc_env.solvers.base.base_solver import BaseSolver
from .arc_env.solvers.base.solver_registry import SolverRegistry
# Placeholder concrete solvers might not be part of main public API unless intended
# from .arc_env.solvers.implementations.heuristic.placeholder_heuristic_solver import PlaceholderHeuristicSolver

# Exceptions (useful for users to catch specific errors)
from .arc_env.exceptions import (
    ARCError, ConfigurationError, DataLoadError, OperationError, TaskNotSetError, InvalidActionError, SolverError
)

# Utilities (selectively expose if they are general purpose for users)
# from .arc_env.utils.logging import setup_logger, get_logger # Usually not part of library's direct API
# from .arc_env.utils.random import set_global_seed # Potentially useful

# Define __all__ to specify what `from arc_env import *` imports.
# It's good practice, though `import *` is often discouraged in production code.
__all__ = [
    "__version__",
    "ARCEnv",
    "MiniARCEnv",
    "MultiTaskARCEnv",
    "BaseARCEnvironment",
    "BaseActionSpace",
    "BaseObservationSpace",
    "ARCActionSpace",
    "ARCObservationSpace",
    "ARCStructuredObservation",
    "BaseConfig",
    "EnvironmentConfig",
    "ActionSpaceConfig",
    "SolverConfig",
    "HeuristicSolverConfig",
    "RLSolverConfig",
    "OperationRegistry",
    "BaseOperation",
    "BaseDataLoader",
    "ARCFileLoader",
    "ARCCombinedFileLoader",
    "ARCDataset",
    "ChallengeDataset",
    "BaseSolver",
    "SolverRegistry",
    "ARCError",
    "ConfigurationError",
    "DataLoadError",
    "OperationError",
    "TaskNotSetError",
    "InvalidActionError",
    "SolverError",
    # Add other key classes/functions intended for direct import from `arc_env`.
]

# Optional: Perform any package-level setup here if needed,
# e.g., setting up a default logger for the package if not handled by individual modules.
# from .arc_env.utils.logging import setup_logger
# setup_logger("arc_env_default_logger", level="WARNING", propagate=False) # Example
# However, it's often better to let applications configure logging.

print(f"arc_env package (version {__version__}) initialized.") # Optional: for debug or verbose import
