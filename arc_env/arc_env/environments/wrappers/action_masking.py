from __future__ import annotations

from typing import Any, Dict, Tuple, Union, Optional, Callable, List, TypeVar # Added TypeVar
import gymnasium as gym
import numpy as np

from .base import BaseARCWrapper
# from arc_env.spaces.action_spaces import ARCActionSpace, DecodedArcOps # If needed for type hints or logic
# from arc_env.core.base_operations import BaseOperation # If checking op properties

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType") # This wrapper does not change action type itself, but might restrict it.

class ActionMaskingWrapper(BaseARCWrapper[ObsType, ActType, ObsType, ActType]):
    """
    A wrapper that can identify invalid actions and penalize them,
    or provide an action mask if the environment supports it.

    The definition of an "invalid action" is domain-specific. For ARC, it could mean:
    - An operation that cannot be applied (e.g., rotating an empty selection).
    - An operation that results in no change to the grid when a change is expected.
    - An operation that violates some constraint (e.g., trying to draw outside canvas,
      though the core env might handle this).

    This wrapper's primary role here, based on the skeleton, is to penalize.
    A more advanced version might also modify the observation space to include an action mask.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        invalid_action_penalty: float = -10.0,
        # Optional: A function to determine if an action is valid for a given observation/state.
        # (observation, decoded_action, env_instance) -> bool
        is_action_valid_fn: Optional[Callable[[ObsType, Any, gym.Env], bool]] = None,
        noop_penalty: Optional[float] = None # Specific penalty if action results in no change
        ):
        super().__init__(env)
        self.invalid_action_penalty = invalid_action_penalty
        self.is_action_valid_fn = is_action_valid_fn
        self.noop_penalty = noop_penalty

        # If the environment's observation space is a Dict and we want to add an action mask,
        # we would modify self.observation_space here.
        # For now, this wrapper only penalizes and doesn't alter the observation space.

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """
        Steps through the environment. If the action is deemed invalid by
        `self._is_action_valid()`, applies a penalty.
        Also checks for no-op actions if `noop_penalty` is set.
        """
        # 1. Determine if action is valid (conceptually)
        # This requires decoding the action if it's complex and having state info.
        # The `_is_valid_action` method needs access to the current observation or state.
        # We get the observation *after* the step. So, validity check should ideally be
        # based on the state *before* the action is taken.
        # This wrapper, as per skeleton, checks *after* the step (implicitly).
        # Let's assume _is_valid_action can be called here, or the logic is embedded.

        # If we have a specific validation function, we need the observation *before* the step.
        # This is problematic with the standard gym.Wrapper.step signature.
        # A common pattern for action masking that needs pre-step observation:
        # - The environment itself computes the action mask and includes it in the observation.
        # - Or, the wrapper stores the last observation.

        # For this implementation, let's assume _is_valid_action can be determined
        # either from the action itself or by trying the action and observing consequences.
        # The skeleton's `_is_valid_action(action)` implies it might not need full current obs.

        # Let's refine: the penalty is applied if the *outcome* of the action is problematic,
        # or if the action can be determined invalid a priori.

        is_valid_priori = True # Assume valid unless proven otherwise
        if self.is_action_valid_fn:
            # To use is_action_valid_fn, we need the *current* observation (pre-step).
            # This wrapper doesn't store it. This implies such a function might be
            # passed to the core environment or a more stateful wrapper.
            # For now, we can't effectively use an is_action_valid_fn that needs pre-step obs.
            # Let's assume _is_valid_action is a simpler check based on action properties only.
            pass # Placeholder for where pre-step validation would go.

        # If we don't have a pre-step validation, we execute the action and then check outcome.
        # The skeleton's structure `obs, _, term, trunc, info = self.env.step(action)` then check
        # suggests that the penalty might be based on `info` or a change in `obs`.

        original_obs = None
        if self.noop_penalty is not None:
            # Need to get current observation to check for no-op. This is tricky.
            # The environment should ideally return this info.
            # A common way is for the env's step() to return an "info" dict
            # that indicates if a no-op occurred.
            # If not, this wrapper would need to be more complex (e.g. call env.render("rgb_array")
            # before and after, or store previous obs, which is not ideal for stateless wrapper).
            # Let's assume the wrapped env's info dict might contain "no_op_occurred": True.
            # Or, we can try to access a copy of the state if the env provides it.
            # For now, we'll rely on info dict or a simple heuristic.
            # This part is difficult to implement generically without assumptions on `env`.
            #
            # A simple heuristic: if the observation is a numpy grid, compare before/after.
            # This requires getting the obs before step, which is not directly available.
            # Let's assume for now that no-op detection is part of the inner env's info.
            pass


        # Execute the action in the wrapped environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Post-step checks:
        # 1. Check info dict for invalid action flags from inner environment
        if info.get("invalid_action_from_env", False): # Assuming inner env might set this
            reward = self.invalid_action_penalty
            info["action_masking_penalty_applied"] = "invalid_action_from_env"

        # 2. Use the skeleton's _is_valid_action. This is unusual as it's called *after* the action.
        # It might be that this method checks properties of the action itself, or it's
        # intended to be used by an agent *before* sending the action (if it had access to this wrapper's method).
        # The skeleton's placement implies it's a check this wrapper performs.
        # Let's assume it's a misplacement in the skeleton and should be a pre-action check,
        # or it's a check on the *type* of action that was just performed,
        # which is less about state validity and more about action format.
        #
        # Given the skeleton's `if not self._is_valid_action(action): reward = -10.0 ...`
        # it implies `_is_valid_action` is called by this wrapper.
        # This would make most sense if `_is_valid_action` can determine validity
        # from `action` alone, or if it's meant to use `self.env`'s current (post-step) state.
        # The latter is strange. Let's assume it's a check on the action itself.

        # Let's follow the skeleton's direct implication:
        # This means `_is_valid_action` is a method of this wrapper.
        if not self._is_valid_action(action, obs, info): # Pass current obs and info for context
            # Apply penalty if _is_valid_action (implemented by user or subclass) returns False.
            # This overrides any reward from the environment for this step.
            reward = self.invalid_action_penalty
            info["action_masking_penalty_applied"] = "wrapper_is_valid_action_check"
            # The skeleton example also sets info["invalid_action"] = True
            info["invalid_action"] = True # Match skeleton

        # 3. Check for no-op if noop_penalty is configured
        if self.noop_penalty is not None and info.get("no_op_occurred", False):
            # Assuming the environment sets "no_op_occurred" in info.
            # If it was an invalid action that also happened to be a no-op,
            # the invalid_action_penalty might take precedence or they could sum.
            # For now, let's assume noop_penalty is separate.
            if not info.get("action_masking_penalty_applied"): # Don't penalize twice if already hit by invalid action
                reward = self.noop_penalty
                info["action_masking_noop_penalty_applied"] = True

        return obs, reward, terminated, truncated, info

    def _is_valid_action(self, action: ActType, current_obs_after_action: ObsType, info_after_action: Dict[str,Any]) -> bool:
        """
        Checks if the taken action is considered valid based on its properties or outcome.
        This method can be overridden by subclasses for specific validation logic.
        The default implementation assumes actions are valid unless other info suggests otherwise.

        Args:
            action: The action that was taken.
            current_obs_after_action: The observation *after* the action was applied.
            info_after_action: The info dict *after* the action was applied.

        Returns:
            True if the action is considered valid by this wrapper's criteria, False otherwise.
        """
        # Default: assume valid. Subclasses or is_action_valid_fn provide specific logic.
        # This method, in the context of the skeleton's step(), is called *after* self.env.step().
        # So, it can be used to evaluate if the *result* of the action implies it was invalid.
        # For example, if info_after_action contains an error message from a lower level.

        # The skeleton example `if not self._is_valid_action(action):` seems to imply
        # that _is_valid_action itself determines validity.
        # This is where custom logic for the specific environment/task would go.
        # E.g., if action is a tuple of operations, check if they are compatible.
        # Or, if the environment state (in current_obs_after_action or self.env)
        # is now in an illegal configuration due to the action.

        # For this placeholder, let's assume it's always true, and relies on info dict flags.
        # A user would override this or provide `is_action_valid_fn` for more complex checks.
        if self.is_action_valid_fn:
            # This is tricky as is_action_valid_fn ideally needs pre-step observation.
            # If it's designed to work with post-step observation:
            # return self.is_action_valid_fn(current_obs_after_action, action, self.env_unwrapped)
            pass # Cannot use it effectively here without pre-step obs.

        # If the info dict already flagged it as invalid from env, this func might not need to run.
        # However, the skeleton implies this wrapper has its own check.
        # Let's make a simple example: if action is outside the defined action space
        # (though this should be caught by the environment itself usually).
        if not self.action_space.contains(action): # The wrapper's action_space
            # This check is usually done by gym.Env itself before its _step.
            # If it reaches here and action is not in space, it's an issue with env's containment check.
            # print(f"Warning: ActionMaskingWrapper._is_valid_action found action {action} not in space {self.action_space}")
            # return False # This would be a fundamental error.
            pass

        # Placeholder: always return True, actual validation logic is specific.
        # The `info.get("invalid_action_from_env", False)` or similar flags from the
        # core environment are more reliable for post-step validation of outcome.
        return True

    # If this wrapper were to provide an action mask in the observation:
    # def observation(self, observation: ObsType) -> Dict[str, Any]: # Assuming new obs is Dict
    #     action_mask = self.compute_action_mask(observation)
    #     # Add action_mask to observation, assuming observation is a dict
    #     if isinstance(observation, dict):
    #         return {**observation, "action_mask": action_mask}
    #     else: # If obs is not dict, how to add mask? Complicates things.
    #         raise TypeError("Cannot add action_mask to non-dict observation.")

    # def compute_action_mask(self, observation: ObsType) -> np.ndarray:
    #     """
    #     Computes a boolean mask indicating valid actions for the current observation.
    #     This requires iterating through all possible actions and checking their validity.
    #     Can be computationally expensive.
    #     """
    #     if not isinstance(self.action_space, gym.spaces.Discrete):
    #         raise NotImplementedError("Action masking is typically implemented for Discrete action spaces.")

    #     mask = np.ones(self.action_space.n, dtype=bool)
    #     for i in range(self.action_space.n):
    #         # This is where self._is_valid_action or a similar pre-step check would be crucial.
    #         # if not self._is_action_valid_for_obs(i, observation): # Needs a pre-step validator
    #         #    mask[i] = False
    #         pass # Placeholder
    #     return mask

# Example Usage:
# if __name__ == "__main__":
#     # Needs a mock environment similar to the one in normalization.py example
#     class MockEnvWithInfo(gym.Env):
#         def __init__(self):
#             self.action_space = gym.spaces.Discrete(3) # Actions 0, 1, 2
#             self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
#             self.current_obs = np.array([0.5], dtype=np.float32)
#         def reset(self, seed=None, options=None):
#             super().reset(seed=seed)
#             self.current_obs = np.array([random.random()], dtype=np.float32)
#             return self.current_obs, {"info": "reset"}
#         def step(self, action):
#             info = {"action_taken": action}
#             reward = 1.0
#             if action == 1: # Let's say action 1 is "invalid" by env's logic
#                 info["invalid_action_from_env"] = True
#                 reward = -5.0 # Env's own penalty
#             elif action == 2: # Let's say action 2 is a "no-op"
#                 info["no_op_occurred"] = True
#                 # No change to self.current_obs for no-op
#             else: # Action 0 is fine, changes obs
#                 self.current_obs = np.array([random.random()], dtype=np.float32)

#             return self.current_obs, reward, False, False, info
#         def render(self): pass
#         def get_challenge_info(self): return {"id": "mock_action_mask_task"}


#     env = MockEnvWithInfo()
#     # Wrap it with ActionMaskingWrapper
#     # Let's define a custom _is_valid_action for the wrapper that considers action 0 invalid
#     class MyCustomActionMask(ActionMaskingWrapper):
#         def _is_valid_action(self, action: int, current_obs_after_action, info_after_action) -> bool:
#             if action == 0: # Wrapper itself considers action 0 invalid.
#                 print("MyCustomActionMask: Action 0 detected as invalid by wrapper.")
#                 return False
#             return True

#     # wrapped_env = ActionMaskingWrapper(env, invalid_action_penalty=-20.0, noop_penalty=-1.0)
#     # Using the custom one:
#     wrapped_env = MyCustomActionMask(env, invalid_action_penalty=-20.0, noop_penalty=-1.0)


#     obs, info = wrapped_env.reset()
#     print(f"Initial obs: {obs}, info: {info}")

#     # Test action 0 (invalid by MyCustomActionMask's _is_valid_action)
#     print("\nTaking action 0 (invalid by wrapper):")
#     obs, reward, term, trunc, info = wrapped_env.step(0)
#     print(f"Obs: {obs}, Reward: {reward}, Term: {term}, Trunc: {trunc}, Info: {info}")
#     # Expected reward: -20.0 (from wrapper), info should indicate penalty.

#     # Test action 1 (invalid by underlying environment)
#     print("\nTaking action 1 (invalid by env):")
#     obs, reward, term, trunc, info = wrapped_env.step(1) # Env returns reward -5.0 and info["invalid_action_from_env"]=True
#     print(f"Obs: {obs}, Reward: {reward}, Term: {term}, Trunc: {trunc}, Info: {info}")
#     # Expected reward: -20.0 (wrapper penalty overrides env's -5.0 if wrapper._is_valid_action also catches it,
#     # or if info["invalid_action_from_env"] triggers the wrapper's penalty).
#     # Current logic: if info["invalid_action_from_env"] is true, wrapper applies penalty.
#     # If MyCustomActionMask._is_valid_action also returns False for action 1, it would re-apply.
#     # The current MyCustomActionMask._is_valid_action only flags action 0. So action 1 is "valid" by wrapper's own check.
#     # So, if inner env sets "invalid_action_from_env", that penalty is applied.
#     # Let's assume the skeleton means the wrapper's own check takes precedence or is primary.
#     # The provided skeleton for ActionMaskingWrapper is:
#     #   if not self._is_valid_action(action):
#     #       obs, _, terminated, truncated, info = self.env.step(action) # This line is problematic, action already taken by agent.
#     #                                                                  # It should be: obs, reward, ... = self.env.step(action)
#     #                                                                  #              if not self._is_valid_action_based_on_outcome_or_action_properties():
#     #                                                                  #                  reward = penalty
#     #       return obs, -10.0, terminated, truncated, {**info, "invalid_action": True}
#     #   return self.env.step(action) # This means env.step is called twice if action is valid by wrapper. Incorrect.
#     # The refined logic in my implementation calls self.env.step(action) once.

#     # Test action 2 (no-op by environment)
#     print("\nTaking action 2 (no-op by env):")
#     obs, reward, term, trunc, info = wrapped_env.step(2) # Env sets info["no_op_occurred"]=True
#     print(f"Obs: {obs}, Reward: {reward}, Term: {term}, Trunc: {trunc}, Info: {info}")
#     # Expected reward: -1.0 (noop_penalty), if not overridden by other invalid checks.
#     # Action 2 is valid by MyCustomActionMask and not flagged by env as "invalid_action_from_env".

#     # Test action that is valid by all criteria (if any action > 2 existed in MockEnvWithInfo)
#     # For MockEnvWithInfo, action 0 is the only one that's "clean" from env perspective.
#     # But MyCustomActionMask makes action 0 invalid.
#     # If we used the base ActionMaskingWrapper (not MyCustomActionMask):
#     # base_wrapped_env = ActionMaskingWrapper(env, invalid_action_penalty=-20.0, noop_penalty=-1.0)
#     # print("\nTaking action 0 (valid by env, base wrapper's _is_valid_action is True):")
#     # obs, reward, term, trunc, info = base_wrapped_env.step(0) # Env gives reward 1.0
#     # print(f"Obs: {obs}, Reward: {reward}, Info: {info}") # Expected reward: 1.0 (no penalties)
