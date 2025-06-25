"""
benchmarking.py

Demonstrates how to use the BenchmarkRunner to evaluate a solver's
performance across a dataset of ARC tasks.
"""
from pathlib import Path
import shutil # For cleanup
import json # For dummy data creation

try:
    from arc_env.environments.arc_env import ARCEnv
    from arc_env.config.environment import EnvironmentConfig
    from arc_env.data.datasets import ARCDataset
    from arc_env.data.loaders.arc_loader import ARCFileLoader
    from arc_env.solvers.base.solver_registry import SolverRegistry
    # Import a placeholder solver for the benchmark
    from arc_env.solvers.implementations.heuristic.placeholder_heuristic_solver import PlaceholderHeuristicSolver
    from arc_env.config.solver import HeuristicSolverConfig
    from arc_env.solvers.evaluation.benchmarking import BenchmarkRunner
    from arc_env.solvers.evaluation.metrics import TaskEvaluationResult # For callback type hint
except ImportError as e:
    print(f"ImportError: {e}. Ensure 'arc_env' is installed or in PYTHONPATH.")
    exit(1)


def create_dummy_benchmark_tasks(base_dir: Path) -> Path:
    """Helper to create a temporary task directory for the benchmark example."""
    tasks_dir = base_dir / "benchmark_arc_tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    task_defs = {
        "bench_task_01": {"train": [{"input": [[1]], "output": [[0]]}],
                          "test": [{"input": [[1,0],[0,1]], "output": [[0,1],[1,0]]}]},
        "bench_task_02": {"train": [{"input": [[2,2]], "output": [[3,3]]}],
                          "test": [{"input": [[4]], "output": [[5]]}]}, # Single test case
        "bench_task_03_multitest": { # Task with multiple test cases
            "train": [{"input": [[6]], "output": [[7]]}],
            "test": [
                {"input": [[8,8]], "output": [[9,9]]}, # Case 1
                {"input": [[1,1,1]], "output": [[0,0,0]]}  # Case 2
            ]
        }
    }
    for task_id, content in task_defs.items():
        with open(tasks_dir / f"{task_id}.json", "w") as f:
            json.dump(content, f)
    return tasks_dir


def benchmark_callback(task_id: str, result: TaskEvaluationResult):
    """Callback function to process results after each task evaluation."""
    print(f"  Benchmark Callback: Task '{task_id}' evaluated.")
    print(f"    Solved: {result.metrics.get('task_solved', 0.0)}")
    print(f"    Test Case Accuracy: {result.metrics.get('test_case_accuracy', 0.0):.2f}")
    # print(f"    Solver Info: {result.solver_info}")


def run_benchmarking_example(task_data_path: Path):
    print("\n--- ARC Environment: Solver Benchmarking Example ---")

    # 1. Setup Data Loader and Dataset
    try:
        loader = ARCFileLoader(data_directory=task_data_path)
        dataset = ARCDataset(data_loader=loader) # No processors for this simple benchmark
    except Exception as e:
        print(f"Error setting up data components: {e}")
        return

    if len(dataset) == 0:
        print("No tasks found in dataset. Aborting benchmark example.")
        return

    print(f"Dataset loaded with {len(dataset)} tasks.")

    # 2. Setup Environment (to be used by BenchmarkRunner)
    # Env config should be compatible with dataset tasks (e.g., canvas size if tasks vary)
    # BenchmarkRunner will use this env instance to run the solver.
    # data_path in env_cfg might be used if env re-initializes its own loader,
    # but BenchmarkRunner passes the dataset directly.
    # It's good practice to ensure env_cfg.data_path is consistent or not needed if loader is explicit.
    env_cfg = EnvironmentConfig(
        canvas_size=15, # A reasonable default canvas for benchmarks
        max_steps=20,   # Max steps per test case attempt by the solver
        data_path=task_data_path # For ARCEnv internal loader if it needs to reload tasks
    )
    try:
        # Pass the same loader to ARCEnv to ensure consistency if it needs to load task details
        # that are not fully captured by ARCTaskData from the dataset.
        env_for_benchmark = ARCEnv(env_config=env_cfg, data_loader=loader)
    except Exception as e:
        print(f"Error initializing ARCEnv for benchmark: {e}")
        return

    # 3. Setup Solver
    # Using the PlaceholderHeuristicSolver for this example.
    # A real benchmark would use more sophisticated solvers.
    solver_registry = SolverRegistry() # If using registry to manage solvers
    solver_registry.register_solver("placeholder_heuristic", PlaceholderHeuristicSolver)

    heuristic_cfg = HeuristicSolverConfig(search_depth=2) # Basic config for the placeholder
    try:
        solver_to_benchmark = solver_registry.create_solver(
            "placeholder_heuristic",
            solver_config=heuristic_cfg,
            action_space=env_for_benchmark.action_space # Solver might need this
        )
    except Exception as e:
        print(f"Error creating solver for benchmark: {e}")
        env_for_benchmark.close(); return

    print(f"Solver '{solver_to_benchmark.__class__.__name__}' initialized for benchmarking.")

    # 4. Setup and Run BenchmarkRunner
    # max_steps_per_test_case in BenchmarkRunner can override env's default for the run.
    runner = BenchmarkRunner(
        dataset=dataset,
        solver=solver_to_benchmark,
        environment=env_for_benchmark,
        per_task_callback=benchmark_callback,
        max_steps_per_test_case=10 # Shorter attempts for this example
    )

    print("\nStarting benchmark run...")
    try:
        summary_results = runner.run_benchmark()
    except Exception as e:
        print(f"Error during benchmark run: {e}")
        env_for_benchmark.close(); return

    # 5. Display Results
    print("\n--- Benchmark Overall Summary ---")
    for key, value in summary_results.items():
        print(f"  {key}: {value}")

    print("\n--- Detailed Results Per Task ---")
    for task_result in runner.get_results():
        print(f"  Task ID: {task_result.task_id}")
        print(f"    Metrics: {task_result.metrics}")
        print(f"    Solver Info: {task_result.solver_info}")
        # print(f"    Predicted Outputs (first one, if any): {task_result.predicted_test_outputs[0][0:3,0:3] if task_result.predicted_test_outputs else 'N/A'}")


    # Cleanup
    env_for_benchmark.close()
    print("\n--- Solver Benchmarking Example Finished ---")


if __name__ == "__main__":
    temp_dir_for_benchmark_ex = Path("_temp_benchmark_example_data")
    example_tasks_path_bench = create_dummy_benchmark_tasks(temp_dir_for_benchmark_ex)

    try:
        run_benchmarking_example(task_data_path=example_tasks_path_bench)
    finally:
        if temp_dir_for_benchmark_ex.exists():
            shutil.rmtree(temp_dir_for_benchmark_ex)
            print(f"\nCleaned up temporary data for benchmark example: {temp_dir_for_benchmark_ex}")
