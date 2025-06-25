from __future__ import annotations

from typing import Dict, Type, Optional, Any, Callable, List

from .base_solver import BaseSolver
from arc_env.config.solver import SolverConfig # For type hinting config
from arc_env.exceptions import ConfigurationError, ARCError, TypeError as CustomTypeError # Added CustomTypeError

# Type for a solver constructor: takes a SolverConfig, returns a BaseSolver instance
SolverConstructor = Callable[[Optional[SolverConfig]], BaseSolver] # Or more specific config type


class SolverRegistry:
    """
    A central registry for ARC solver types.

    This allows dynamic registration and instantiation of different solver
    implementations based on a configuration or name.
    """

    def __init__(self) -> None:
        # Stores solver classes: _solvers[solver_name_str] = SolverClass
        self._solver_classes: Dict[str, Type[BaseSolver]] = {}
        # Alternatively, store constructors if instantiation is more complex:
        # self._solver_constructors: Dict[str, SolverConstructor] = {}

    def register_solver(
        self,
        name: str,
        solver_class: Type[BaseSolver],
        exist_ok: bool = False
    ) -> None:
        """
        Registers a solver class with a given name.

        Args:
            name: The unique name to identify this solver type (e.g., "random", "heuristic_bfs").
            solver_class: The class (subclass of BaseSolver) to register.
            exist_ok: If True, re-registering a solver with the same name will not
                      raise an error (it will be overwritten). Default is False.

        Raises:
            TypeError: If solver_class is not a subclass of BaseSolver.
            ConfigurationError: If the name is already registered and exist_ok is False.
        """
        if not issubclass(solver_class, BaseSolver):
            raise CustomTypeError(f"Solver class '{solver_class.__name__}' must be a subclass of BaseSolver.")

        if not exist_ok and name in self._solver_classes:
            raise ConfigurationError(
                f"Solver type '{name}' already registered. Set exist_ok=True to overwrite."
            )

        self._solver_classes[name] = solver_class
        print(f"SolverRegistry: Registered solver type '{name}' -> {solver_class.__name__}")

    def get_solver_class(self, name: str) -> Optional[Type[BaseSolver]]:
        """
        Retrieves a registered solver class by its name.

        Args:
            name: The name of the solver type.

        Returns:
            The solver class if found, otherwise None.
        """
        return self._solver_classes.get(name)

    def create_solver(
        self,
        name: str,
        solver_config: Optional[SolverConfig] = None,
        # Optional additional args that might be needed by specific solver constructors
        # beyond what's in SolverConfig (e.g., reference to environment's action space).
        **kwargs: Any
    ) -> BaseSolver:
        """
        Creates an instance of a registered solver.

        Args:
            name: The name of the solver type to instantiate.
            solver_config: The configuration object for the solver. This should be an
                           instance of SolverConfig or a subclass specific to the solver.
            **kwargs: Additional keyword arguments to pass to the solver's constructor.
                      These are passed *after* solver_config.

        Returns:
            An instance of the specified BaseSolver subclass.

        Raises:
            ConfigurationError: If the solver name is not registered.
            Exception: Any exception raised by the solver's constructor.
        """
        solver_class = self.get_solver_class(name)
        if not solver_class:
            raise ConfigurationError(
                f"Solver type '{name}' not found in registry. Available: {self.list_available_solvers()}"
            )

        try:
            # Solver constructor should ideally take SolverConfig as first arg (or compatible dict).
            # If kwargs are provided, they are passed too.
            # Ensure solver_config is passed correctly.
            # If solver_class.__init__ expects `solver_config=...`, then:
            # return solver_class(solver_config=solver_config, **kwargs)
            # If it expects config as a dict:
            # config_dict = solver_config.to_dict() if solver_config else {}
            # return solver_class(config_dict, **kwargs)

            # Let's assume constructor is like: __init__(self, solver_config: Optional[SolverConfig], **other_kwargs_for_solver)
            # The BaseSolver.__init__ takes solver_config (which can be a dict).
            # If SolverConfig object is passed, it will be used.
            # If solver_config is None, an empty dict is passed to BaseSolver.

            # If the concrete solver's __init__ has a more specific type hint for solver_config
            # (e.g., HeuristicSolverConfig), the passed solver_config object should match that.
            # The registry doesn't enforce this sub-type matching; the solver class will.
            return solver_class(solver_config=solver_config, **kwargs)
        except Exception as e:
            # Catch TypeErrors from mismatched config types, ValueErrors from validation, etc.
            raise ARCError(
                f"Failed to create solver '{name}' of class {solver_class.__name__} "
                f"with config {type(solver_config).__name if solver_config else None} and kwargs {kwargs}. Error: {e}"
            ) from e


    def list_available_solvers(self) -> List[str]:
        """Returns a list of names of all registered solver types."""
        return sorted(list(self._solver_classes.keys()))


# Global instance (Singleton pattern can be used if preferred, but module-level instance is common)
# solver_registry = SolverRegistry()

# Decorator for easier registration (optional)
# def register_solver_decorator(name: str, registry_instance: SolverRegistry = solver_registry):
#     def decorator(cls: Type[BaseSolver]):
#         registry_instance.register_solver(name, cls)
#         return cls
#     return decorator

# Example Usage:
# if __name__ == "__main__":
#     # Example Solver Classes (would be in separate files)
#     class MyRandomSolver(BaseSolver):
#         def __init__(self, solver_config: Optional[SolverConfig] = None, action_space=None): # action_space via kwargs
#             super().__init__(solver_config.to_dict() if solver_config else None) # BaseSolver expects dict or None
#             if action_space is None: raise ValueError("MyRandomSolver requires action_space kwarg.")
#             self.action_space = action_space
#         def predict_action(self, observation: Any, env_info: Optional[Dict[str, Any]] = None) -> Any:
#             return self.action_space.sample()

#     class MyHeuristicSolver(BaseSolver):
#         def __init__(self, solver_config: Optional[SolverConfig] = None, specific_param: str = "default"):
#             super().__init__(solver_config.to_dict() if solver_config else None)
#             self.specific_param = solver_config.hyperparameters.get("specific_param", specific_param) if solver_config and solver_config.hyperparameters else specific_param # type: ignore
#             print(f"MyHeuristicSolver initialized with specific_param: {self.specific_param}")
#         def predict_action(self, observation: Any, env_info: Optional[Dict[str, Any]] = None) -> Any:
#             return 0 # Dummy action

#     # Test Registry
#     test_registry = SolverRegistry()
#     test_registry.register_solver("random_test", MyRandomSolver)
#     test_registry.register_solver("heuristic_test", MyHeuristicSolver)

#     print("Available solvers:", test_registry.list_available_solvers())

#     # Create instances
#     try:
#         # For RandomSolver, it needs action_space passed via kwargs
#         import gymnasium as gym # For dummy action space
#         dummy_as = gym.spaces.Discrete(3)
#         random_solver_instance = test_registry.create_solver("random_test", solver_config=None, action_space=dummy_as)
#         print(f"Created random_test solver: {type(random_solver_instance).__name__}")
#         print(f"  Action from random solver: {random_solver_instance.predict_action(None)}")

#         # For HeuristicSolver, pass a config
#         heuristic_cfg_obj = SolverConfig(solver_type="heuristic_test", hyperparameters={"specific_param": "custom_value"})
#         heuristic_solver_instance = test_registry.create_solver("heuristic_test", solver_config=heuristic_cfg_obj)
#         print(f"Created heuristic_test solver: {type(heuristic_solver_instance).__name__}")
#         # Accessing internal param to check if config was used (heuristic_solver_instance.specific_param)

#         # Create with default specific_param (no config or config without that hyperparam)
#         heuristic_solver_default = test_registry.create_solver("heuristic_test", solver_config=SolverConfig())

#         # Try to create non-existent solver
#         # test_registry.create_solver("non_existent_solver") # Raises ConfigurationError

#     except Exception as e:
#         print(f"Error during solver creation example: {e}")
#         import traceback
#         traceback.print_exc()
