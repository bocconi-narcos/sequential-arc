"""
check_solvers.py

A utility script to perform basic checks or validations on implemented solvers.
This might involve:
- Ensuring solvers can be instantiated via the registry.
- Running a solver on a simple dummy task to see if predict_action works.
- Checking for required methods or configurations.
"""
import argparse
from pathlib import Path
import sys

# Adjust path to import from arc_env if script is run from outside the package root
# This is a common pattern for scripts.
# Alternatively, ensure arc_env is installed (`pip install -e .`) and then imports work directly.
# SCRIPT_DIR = Path(__file__).resolve().parent
# PACKAGE_ROOT = SCRIPT_DIR.parent # Assuming scripts/ is one level down from arc_env package root
# sys.path.insert(0, str(PACKAGE_ROOT.parent)) # Add project root to path

try:
    from arc_env.solvers.base.solver_registry import SolverRegistry
    from arc_env.solvers.implementations.heuristic.placeholder_heuristic_solver import PlaceholderHeuristicSolver
    from arc_env.solvers.implementations.rl.placeholder_rl_solver import PlaceholderRLSolver
    from arc_env.solvers.implementations.hybrid.placeholder_hybrid_solver import PlaceholderHybridSolver
    from arc_env.config.solver import SolverConfig, HeuristicSolverConfig, RLSolverConfig
    # For testing predict_action, we might need a dummy env or action space
    import gymnasium as gym
except ImportError as e:
    print(f"ImportError: {e}. Failed to import arc_env components.")
    print("Please ensure 'arc_env' is installed (e.g., `pip install -e .` from project root) "
          "or PYTHONPATH is configured correctly.")
    sys.exit(1)

def register_placeholder_solvers(registry: SolverRegistry):
    """Helper to register known placeholder solvers for checking."""
    registry.register_solver("placeholder_heuristic", PlaceholderHeuristicSolver, exist_ok=True)
    registry.register_solver("placeholder_rl", PlaceholderRLSolver, exist_ok=True)
    registry.register_solver("placeholder_hybrid", PlaceholderHybridSolver, exist_ok=True)
    print("Registered placeholder solvers for checking.")

def check_solver_instantiation(registry: SolverRegistry, solver_name: str, config: Optional[SolverConfig] = None, **kwargs):
    """Attempts to instantiate a solver and reports success/failure."""
    print(f"\n--- Checking Solver: {solver_name} ---")
    try:
        solver_instance = registry.create_solver(solver_name, solver_config=config, **kwargs)
        print(f"  SUCCESS: Instantiated '{solver_name}' (type: {type(solver_instance).__name__})")

        # Optional: Perform a basic predict_action call if possible
        # This requires a dummy observation and info.
        if hasattr(solver_instance, "predict_action"):
            dummy_obs = {} # Placeholder observation
            dummy_info = {}
            try:
                action = solver_instance.predict_action(dummy_obs, dummy_info)
                print(f"  SUCCESS: predict_action() returned: {action} (type: {type(action).__name__})")
            except Exception as e_predict:
                print(f"  WARNING: predict_action() failed for '{solver_name}': {e_predict}")
        else:
            print(f"  INFO: Solver '{solver_name}' does not have a predict_action method (or not as expected).")

    except Exception as e_create:
        print(f"  FAILURE: Could not instantiate or test '{solver_name}': {e_create}")

def main(args):
    """Main function to run solver checks."""
    registry = SolverRegistry()
    register_placeholder_solvers(registry)

    # Dummy action/observation spaces for solvers that might need them during init or predict
    dummy_action_space = gym.spaces.Discrete(5)
    dummy_observation_space = gym.spaces.Box(low=0, high=9, shape=(3,3), dtype=np.uint8)


    if args.solver_name:
        if args.solver_name not in registry.list_available_solvers():
            print(f"Error: Solver '{args.solver_name}' is not registered. Available: {registry.list_available_solvers()}")
            return
        solvers_to_check = [args.solver_name]
    else:
        solvers_to_check = registry.list_available_solvers()
        if not solvers_to_check:
            print("No solvers registered to check.")
            return
        print(f"Checking all registered solvers: {solvers_to_check}")

    for name in solvers_to_check:
        # Provide default configs or specific ones if needed for instantiation test
        config_to_use: Optional[SolverConfig] = None
        extra_kwargs = {}

        if name == "placeholder_heuristic":
            config_to_use = HeuristicSolverConfig()
            extra_kwargs["action_space"] = dummy_action_space
        elif name == "placeholder_rl":
            config_to_use = RLSolverConfig()
            extra_kwargs["action_space"] = dummy_action_space
            extra_kwargs["observation_space"] = dummy_observation_space
        elif name == "placeholder_hybrid":
            config_to_use = SolverConfig(solver_type="placeholder_hybrid") # Basic config
            extra_kwargs["action_space"] = dummy_action_space
        else: # For other solvers, use a generic SolverConfig or None
            config_to_use = SolverConfig(solver_type=name)

        check_solver_instantiation(registry, name, config=config_to_use, **extra_kwargs)

    print("\n--- Solver Checks Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check and validate ARC solvers.")
    parser.add_argument(
        "--solver_name",
        type=str,
        default=None,
        help="Name of a specific solver to check. If None, checks all registered solvers."
    )
    # Add more arguments if needed, e.g., path to solver configs, specific checks to run.

    parsed_args = parser.parse_args()
    main(parsed_args)

"""
Example usage from project root (after `pip install -e .`):

# Check all registered placeholder solvers
python arc_env/scripts/check_solvers.py

# Check a specific solver
python arc_env/scripts/check_solvers.py --solver_name placeholder_heuristic
"""
