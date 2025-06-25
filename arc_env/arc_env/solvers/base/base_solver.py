from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from arc_env.config.solver import SolverConfig # For typed config, assuming no circularity risk here
# from arc_env.spaces.action_spaces import ARCActionSpace # Avoid for now if potential circularity
# from arc_env.spaces.observation_spaces import ARCStructuredObservation # Avoid for now

# Using Any for observation and action return types to keep BaseSolver generic.
# Concrete solvers can use more specific types from arc_env.spaces if needed.

class BaseSolver(ABC):
    """
    Abstract base class for all ARC solvers.

    A solver is responsible for observing the environment state (an ARC task)
    and predicting an action or a sequence of actions to solve the task.
    """

    def __init__(self, solver_config: Optional[Dict[str, Any]] = None): # Change to SolverConfig type later
        """
        Initializes the solver.

        Args:
            solver_config: An optional SolverConfig object or dictionary containing
                           configuration parameters for the solver (e.g., hyperparameters,
                           model paths).
        """
        self.config = solver_config if solver_config is not None else {}
        self._validate_config()
        self.current_task_id: Optional[str] = None
        self.current_task_info: Optional[Dict[str, Any]] = None # From env.get_challenge_info()

    def _validate_config(self) -> None:
        """
        Validates the solver's configuration.
        Subclasses should override this to check for their specific parameters.
        """
        # Example:
        # if "expected_hyperparam" not in self.config:
        #     raise ConfigurationError("Missing 'expected_hyperparam' in solver config.")
        pass # Default: no specific validation

    @abstractmethod
    def predict_action(
        self,
        observation: Any, # Should be ARCStructuredObservation
        env_info: Optional[Dict[str, Any]] = None # Info from env.reset() or env.step()
    ) -> Any: # Should be action compatible with ARCActionSpace (e.g. int or Dict)
        """
        Predicts the next action based on the current observation and environment state.

        Args:
            observation: The current observation from the ARC environment.
                         Type should align with ARCObservationSpace.
            env_info: Optional dictionary containing auxiliary information from the
                      environment, which might include current task ID, step count, etc.
                      This can be crucial for stateful solvers or those that adapt per task.

        Returns:
            An action compatible with the environment's action space.
        """
        pass

    def set_current_task_context(self, task_id: str, task_info: Dict[str, Any]) -> None:
        """
        Provides the solver with context about the current task it's trying to solve.
        This can be called by the training loop or evaluation script when a new task starts.

        Args:
            task_id: The identifier of the current task.
            task_info: A dictionary of information about the task (e.g., from env.get_challenge_info()).
        """
        if self.current_task_id != task_id: # If task has changed or first time
            self.current_task_id = task_id
            self.current_task_info = task_info
            self.reset_for_new_task() # Call specific reset logic for task change

    def reset_for_new_task(self) -> None:
        """
        Resets any internal state of the solver that is specific to a single task.
        Called when `set_current_task_context` indicates a new task.
        Subclasses should override this if they maintain task-specific state.
        """
        # Example: Clear cached solutions for previous task, reset search tree, etc.
        pass

    def episode_reset(self) -> None:
        """
        Resets any internal state of the solver at the beginning of a new episode
        (e.g., when env.reset() is called). This is for episode-level state,
        which might be different from task-level reset if a task involves multiple episodes
        or attempts.
        By default, could call reset_for_new_task if task context isn't managed more granularly.
        """
        # For many solvers, this might be similar to reset_for_new_task or a no-op
        # if all relevant state is passed in predict_action's observation/env_info.
        pass

    # Optional methods for solvers that involve training or saving/loading models:
    # def train(self, dataset: Any) -> None:
    #     """Trains the solver on a given dataset (e.g., a list of ARCTaskData)."""
    #     raise NotImplementedError(f"{self.__class__.__name__} does not support training.")

    # def save(self, filepath: str) -> None:
    #     """Saves the solver's state (e.g., model weights) to a file."""
    #     raise NotImplementedError(f"{self.__class__.__name__} does not support saving.")

    # def load(self, filepath: str) -> None:
    #     """Loads the solver's state from a file."""
    #     raise NotImplementedError(f"{self.__class__.__name__} does not support loading.")


# Example of a very simple concrete solver (for illustration):
# class RandomSolver(BaseSolver):
#     def __init__(self, action_space: gym.Space, solver_config: Optional[Dict[str, Any]] = None):
#         super().__init__(solver_config)
#         self.action_space = action_space # Needs the env's action space to sample

#     def predict_action(self, observation: Any, env_info: Optional[Dict[str, Any]] = None) -> Any:
#         return self.action_space.sample()

#     def reset_for_new_task(self) -> None:
#         print(f"RandomSolver: Reset for task {self.current_task_id or 'Unknown'}")
#         pass

# if __name__ == '__main__':
#     import gymnasium as gym
#     # Dummy config and action space for testing
#     dummy_action_space = gym.spaces.Discrete(5)
#     random_s = RandomSolver(action_space=dummy_action_space)

#     # Simulate task context
#     random_s.set_current_task_context("task_abc", {"num_train": 2, "num_test":1})

#     # Simulate getting an action
#     dummy_obs = {"grid": [[0,1],[1,0]]} # Simplified observation
#     action = random_s.predict_action(dummy_obs)
#     print(f"RandomSolver predicted action: {action} (should be in [0,4])")

#     random_s.episode_reset()
#     print("RandomSolver episode reset called.")
