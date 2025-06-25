from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal

from .base import BaseConfig

@dataclass
class SolverConfig(BaseConfig):
    """
    Base configuration for an ARC solver.

    Attributes:
        solver_type: The type or name of the solver to be used (e.g., "heuristic", "rl_ppo", "hybrid_search").
                     This would typically map to a registered solver implementation.
        solver_id: A unique identifier for this specific solver configuration instance.
                   Default is "default_solver".
        hyperparameters: A dictionary to store solver-specific hyperparameters.
                         The structure of this dict will vary greatly depending on the solver_type.
        load_path: Optional path to load a pre-trained solver model or saved state.
        log_level: Logging level for the solver (e.g., "INFO", "DEBUG").
    """
    solver_type: str = "default_heuristic" # Example: "heuristic_bfs", "rl_ppo", "custom_genetic"
    solver_id: str = "default_solver"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    load_path: Optional[str] = None # Path to a saved model/state
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    def validate(self) -> None:
        """Validate solver configuration parameters."""
        super().validate()

        if not self.solver_type:
            raise ValueError("solver_type must be specified.")

        if not self.solver_id:
            raise ValueError("solver_id must be specified.")

        if self.load_path and not isinstance(self.load_path, str): # Basic check
            raise ValueError("load_path must be a string (filepath) if provided.")

        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}. Must be one of {valid_log_levels}.")

        # Further validation of hyperparameters would typically be done by the specific
        # solver class once it's instantiated with this config, as only the solver
        # knows what hyperparameters it expects.
        # For example, a HeuristicSolverConfig might check for 'max_depth'.
        # An RLSolverConfig might check for 'learning_rate'.

# Example of more specific solver configurations inheriting from SolverConfig:

@dataclass
class HeuristicSolverConfig(SolverConfig):
    """Configuration for a heuristic-based solver."""
    solver_type: str = field(default="heuristic_search", init=False) # Override default type
    search_depth: int = 5
    time_limit_ms: Optional[int] = 10000 # Time limit in milliseconds

    def validate(self) -> None:
        super().validate()
        if self.search_depth <= 0:
            raise ValueError("search_depth must be positive.")
        if self.time_limit_ms is not None and self.time_limit_ms <= 0:
            raise ValueError("time_limit_ms must be positive if set.")

@dataclass
class RLSolverConfig(SolverConfig):
    """Configuration for an RL-based solver."""
    solver_type: str = field(default="rl_agent", init=False) # Override default type
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    model_architecture: str = "default_cnn" # e.g., "resnet_small", "custom_transformer"

    def validate(self) -> None:
        super().validate()
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        # model_architecture validation might involve checking against a registry of known architectures.

# Example Usage:
# if __name__ == "__main__":
#     try:
#         # Generic Solver Config
#         base_solver_conf = SolverConfig(
#             solver_type="some_custom_solver",
#             solver_id="my_solver_001",
#             hyperparameters={"param1": 10, "param2": "abc"}
#         )
#         print(f"Base Solver Config: {base_solver_conf.to_dict()}")
#         # base_solver_conf.validate() # Called in __post_init__

#         # Heuristic Solver Config
#         heuristic_conf = HeuristicSolverConfig(
#             solver_id="heuristic_bfs_test",
#             search_depth=3,
#             time_limit_ms=5000
#         )
#         print(f"Heuristic Solver Config: {heuristic_conf.to_dict()}")
#         # heuristic_conf.validate()

#         # RL Solver Config
#         rl_conf = RLSolverConfig(
#             solver_id="ppo_agent_run1",
#             learning_rate=5e-5,
#             batch_size=64,
#             model_architecture="transformer_small"
#         )
#         print(f"RL Solver Config: {rl_conf.to_dict()}")
#         # rl_conf.validate()

#         # Example of validation error
#         # invalid_heuristic = HeuristicSolverConfig(search_depth=0)

#         # Example of using hyperparameters in the base SolverConfig for a type not predefined
#         genetic_solver_conf = SolverConfig(
#             solver_type="genetic_algorithm",
#             solver_id="ga_population_test",
#             hyperparameters={
#                 "population_size": 100,
#                 "mutation_rate": 0.01,
#                 "crossover_rate": 0.8,
#                 "generations": 50
#             }
#         )
#         print(f"Genetic Solver Config: {genetic_solver_conf.to_dict()}")


#     except ValueError as e:
#         print(f"Configuration Error: {e}")
