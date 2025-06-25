from __future__ import annotations

import time
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from arc_env.data.datasets import ARCDataset # To iterate over tasks
from arc_env.solvers.base.base_solver import BaseSolver
from arc_env.environments.arc_env import ARCEnv # To run the solver in
from .metrics import TaskEvaluationResult, evaluate_single_task_attempt
from arc_env.exceptions import ARCError, TaskNotSetError # Added TaskNotSetError


class BenchmarkRunner:
    """
    Runs a solver over a dataset of ARC tasks and collects evaluation results.
    """

    def __init__(
        self,
        dataset: ARCDataset,
        solver: BaseSolver,
        # The environment instance used for running the solver.
        # It should be configured compatibly with the solver and dataset.
        environment: ARCEnv,
        # Optional: callback function called after each task is evaluated
        # (task_id, TaskEvaluationResult) -> None
        per_task_callback: Optional[Callable[[str, TaskEvaluationResult], None]] = None,
        # Max steps per test case attempt by the solver in the environment
        max_steps_per_test_case: Optional[int] = None
    ):
        if not isinstance(dataset, ARCDataset):
            raise TypeError("dataset must be an instance of ARCDataset.")
        if not isinstance(solver, BaseSolver): # Check protocol/ABC
            raise TypeError("solver must be an instance of BaseSolver.")
        if not isinstance(environment, ARCEnv):
            raise TypeError("environment must be an instance of ARCEnv.")

        self.dataset = dataset
        self.solver = solver
        self.env = environment # The env instance will be used to simulate solver's actions
        self.per_task_callback = per_task_callback

        # If max_steps_per_test_case is provided here, it could override env's config for benchmark run.
        # This requires env to support setting max_steps dynamically or via reset options.
        # For now, assume env.env_config.max_steps is used, or this is for external loop control.
        self.max_steps_per_test_case = max_steps_per_test_case
        if self.max_steps_per_test_case is None and self.env.env_config.max_steps:
            self.max_steps_per_test_case = self.env.env_config.max_steps
        elif self.max_steps_per_test_case is None: # No limit from env or runner
            self.max_steps_per_test_case = float('inf') # Effectively no limit by runner

        self.overall_results: List[TaskEvaluationResult] = []
        self.benchmark_summary: Dict[str, Any] = {}


    def run_benchmark(self) -> Dict[str, Any]:
        """
        Executes the solver on all tasks in the dataset and aggregates results.

        Returns:
            A summary dictionary of the benchmark results.
        """
        self.overall_results = []
        total_tasks = len(self.dataset)
        start_time_benchmark = time.time()

        print(f"Starting benchmark for solver '{self.solver.__class__.__name__}' on {total_tasks} tasks.")

        for i, task_data_item in enumerate(self.dataset): # ARCDataset yields ARCDatasetItem (ARCTaskData)
            task_id = task_data_item.task_id # type: ignore # ARCTaskData protocol has optional task_id
            if not task_id:
                task_id = f"unnamed_task_{i}"
                print(f"Warning: Task at index {i} has no task_id. Using generated ID: {task_id}")

            print(f"\nEvaluating task {i+1}/{total_tasks}: {task_id}")

            # Provide task context to solver (e.g. from env.get_challenge_info after setting task)
            # This requires setting the task in the env first.
            try:
                self.env.set_task(task_id) # Load task into the benchmark environment
                challenge_info = self.env.get_challenge_info()
                self.solver.set_current_task_context(task_id, challenge_info)
            except Exception as e:
                print(f"Error setting up task '{task_id}' for solver: {e}. Skipping.")
                # Create a dummy error result for this task
                error_res = TaskEvaluationResult(task_id)
                error_res.solver_info["error"] = f"Setup failed: {e}"
                error_res.add_metric("task_solved", 0.0) # Mark as unsolved
                self.overall_results.append(error_res)
                if self.per_task_callback: self.per_task_callback(task_id, error_res)
                continue

            predicted_grids_for_task: List[np.ndarray] = []
            target_grids_for_task: List[np.ndarray] = []

            num_test_cases_in_task = len(task_data_item.test) # type: ignore
            task_total_steps = 0
            task_total_time_ms = 0
            task_failed_critically = False

            for test_idx in range(num_test_cases_in_task):
                print(f"  Attempting test case {test_idx + 1}/{num_test_cases_in_task} for task '{task_id}'...")
                # Reset environment to this specific test case of the current task
                # ARCEnv.reset() options can take 'test_input_idx'
                try:
                    # Seed could be fixed per test case for reproducibility if desired
                    obs, info = self.env.reset(options={"test_input_idx": test_idx})
                except Exception as e:
                    print(f"    Error resetting env for task '{task_id}', test case {test_idx}: {e}. Skipping test case.")
                    # Add a placeholder for this failed test case's prediction
                    # Needs target grid shape if possible, or use a special marker.
                    # For now, this means this test case will not be "solved".
                    # If target_grids_for_task is populated, this will lead to mismatch later.
                    # Better: try to get target grid and add a dummy prediction.
                    try:
                        _, target_g = self.env._get_current_task_test_pair() # Access internal method for target
                        target_grids_for_task.append(np.copy(target_g))
                        predicted_grids_for_task.append(np.array([[-1]])) # Dummy error grid
                    except Exception: # If even getting target fails
                         pass # This test case will be effectively skipped in metrics
                    task_failed_critically = True; break # Stop processing this task

                self.solver.episode_reset() # Reset solver state for new episode/test case

                start_time_case = time.time()
                current_case_steps = 0

                final_prediction_for_case: Optional[np.ndarray] = None

                for step_num in range(int(self.max_steps_per_test_case)): # type: ignore
                    current_case_steps += 1
                    action = self.solver.predict_action(obs, info)

                    try:
                        obs, reward, terminated, truncated, info = self.env.step(action)
                    except Exception as e:
                        print(f"    Error during env.step for task '{task_id}', test {test_idx}: {e}. Ending attempt.")
                        task_failed_critically = True; break # Critical error in env

                    if terminated: # Task solved for this test case
                        print(f"    Test case {test_idx+1} SOLVED in {current_case_steps} steps.")
                        final_prediction_for_case = np.copy(self.env.current_grid) # Grid that solved it
                        break
                    if truncated: # Max steps reached for this test case
                        print(f"    Test case {test_idx+1} TRUNCATED after {current_case_steps} steps (max: {self.max_steps_per_test_case}).")
                        final_prediction_for_case = np.copy(self.env.current_grid) # Grid at truncation
                        break

                if task_failed_critically: break # From inner env error loop

                end_time_case = time.time()
                task_total_time_ms += (end_time_case - start_time_case) * 1000
                task_total_steps += current_case_steps

                if final_prediction_for_case is None: # Loop finished without term/trunc (should not happen if max_steps is reasonable)
                    print(f"    Warning: Test case {test_idx+1} loop finished without termination/truncation. Using current grid.")
                    final_prediction_for_case = np.copy(self.env.current_grid)

                predicted_grids_for_task.append(final_prediction_for_case)
                # Target grid is from the current state of env after reset for this test_idx
                # self.env.target_grid should be correctly set.
                if self.env.target_grid is not None:
                    target_grids_for_task.append(np.copy(self.env.target_grid))
                else: # Should not happen if env reset correctly
                    print(f"    Error: Target grid not available in env after reset for test case {test_idx}. Metrics may be affected.")
                    # Add a dummy target to maintain list length parity if needed, or handle in metrics.
                    target_grids_for_task.append(np.array([[-2]])) # Special error marker grid

            # After all test cases for the task (or critical failure)
            solver_run_info_for_task = {
                "total_steps": task_total_steps,
                "total_time_ms": round(task_total_time_ms, 3),
                "critically_failed": task_failed_critically
            }
            task_eval_result = evaluate_single_task_attempt(
                task_id, predicted_grids_for_task, target_grids_for_task, solver_run_info_for_task
            )
            self.overall_results.append(task_eval_result)

            if self.per_task_callback:
                self.per_task_callback(task_id, task_eval_result)

        # Benchmark finished, calculate summary
        end_time_benchmark = time.time()
        self.benchmark_summary = self._calculate_summary_stats()
        self.benchmark_summary["total_benchmark_time_s"] = round(end_time_benchmark - start_time_benchmark, 3)

        print(f"\nBenchmark finished. Total time: {self.benchmark_summary['total_benchmark_time_s']:.2f}s")
        print(f"Summary: {self.benchmark_summary}")
        return self.benchmark_summary

    def _calculate_summary_stats(self) -> Dict[str, Any]:
        if not self.overall_results:
            return {"message": "No tasks evaluated."}

        num_tasks_evaluated = len(self.overall_results)
        num_tasks_fully_solved = sum(res.metrics.get("task_solved", 0.0) for res in self.overall_results)

        avg_test_case_accuracy = np.mean([res.metrics.get("test_case_accuracy", 0.0) for res in self.overall_results])
        avg_pixelwise_accuracy = np.mean([res.metrics.get("pixelwise_accuracy", 0.0) for res in self.overall_results])

        total_steps_across_all = sum(res.solver_info.get("total_steps", 0) for res in self.overall_results)
        total_time_ms_across_all = sum(res.solver_info.get("total_time_ms", 0) for res in self.overall_results)

        num_critical_failures = sum(1 for res in self.overall_results if res.solver_info.get("critically_failed", False))

        summary = {
            "solver_name": self.solver.__class__.__name__,
            "dataset_size": len(self.dataset.task_ids) if self.dataset else 0, # Original dataset size before filtering in runner
            "num_tasks_attempted": num_tasks_evaluated,
            "num_tasks_fully_solved": int(num_tasks_fully_solved),
            "overall_solve_rate": float(num_tasks_fully_solved) / num_tasks_evaluated if num_tasks_evaluated > 0 else 0.0,
            "average_test_case_accuracy": round(avg_test_case_accuracy, 4),
            "average_pixelwise_accuracy": round(avg_pixelwise_accuracy, 4),
            "total_steps_all_tasks": total_steps_across_all,
            "total_time_ms_all_tasks": round(total_time_ms_across_all, 3),
            "average_steps_per_task": round(float(total_steps_across_all) / num_tasks_evaluated, 2) if num_tasks_evaluated > 0 else 0.0,
            "average_time_ms_per_task": round(float(total_time_ms_across_all) / num_tasks_evaluated, 3) if num_tasks_evaluated > 0 else 0.0,
            "num_critical_task_failures": num_critical_failures,
        }
        return summary

    def get_results(self) -> List[TaskEvaluationResult]:
        return self.overall_results

    def get_summary(self) -> Dict[str, Any]:
        return self.benchmark_summary


# Example Usage (requires many components to be set up)
# if __name__ == '__main__':
#     from arc_env.data.loaders.arc_loader import ARCFileLoader
#     from arc_env.config.environment import EnvironmentConfig
#     from arc_env.solvers.implementations.heuristic.placeholder_heuristic_solver import PlaceholderHeuristicSolver # Example solver
#     from arc_env.config.solver import HeuristicSolverConfig

#     # 1. Setup dummy data and loader
#     dummy_bench_task_dir = Path("temp_benchmark_tasks")
#     dummy_bench_task_dir.mkdir(exist_ok=True)
#     task_defs = {
#         "b_task_01": {"train": [{"input": [[1]], "output": [[0]]}], "test": [{"input": [[1]], "output": [[0]]}]}, # Solvable by copying input
#         "b_task_02": {"train": [{"input": [[2]], "output": [[3]]}], "test": [{"input": [[2]], "output": [[4]]}]}  # Different output
#     }
#     import json
#     for tid, content in task_defs.items():
#         with open(dummy_bench_task_dir / f"{tid}.json", "w") as f: json.dump(content, f)
#     loader = ARCFileLoader(data_directory=dummy_bench_task_dir)
#     dataset = ARCDataset(data_loader=loader)

#     # 2. Setup Environment (needs to load tasks from the same place or be compatible)
#     env_cfg = EnvironmentConfig(data_path=str(dummy_bench_task_dir), max_steps=10) # Env needs data_path for its own loader if used
#     env = ARCEnv(env_config=env_cfg, data_loader=loader) # Pass same loader to env

#     # 3. Setup Solver
#     solver_cfg = HeuristicSolverConfig() # Default config for placeholder
#     # The placeholder heuristic solver samples actions. It might solve by chance.
#     # For a predictable test, one might need a solver that, e.g., always outputs the input grid.
#     class CopyInputSolver(BaseSolver): # A simple solver for testing benchmark
#         def predict_action(self, obs, info=None): return 0 # Dummy action, assumes env starts with input as current_grid
#         def __init__(self, solver_config=None, **kwargs): super().__init__(solver_config)
#         # This solver doesn't take actions to change grid; it would need to submit current_grid.
#         # The benchmark loop expects actions. Let's use PlaceholderHeuristicSolver.

#     solver = PlaceholderHeuristicSolver(solver_config=solver_cfg, action_space=env.action_space)

#     # 4. Setup BenchmarkRunner
#     def my_callback(task_id, result):
#         print(f"  Callback for {task_id}: Solved = {result.metrics.get('task_solved')}")

#     runner = BenchmarkRunner(
#         dataset=dataset,
#         solver=solver,
#         environment=env,
#         per_task_callback=my_callback,
#         max_steps_per_test_case=5 # Override env's max_steps for this run
#     )

#     try:
#         print("--- Running Benchmark ---")
#         summary = runner.run_benchmark()

#         print("\n--- Benchmark Detailed Results ---")
#         for res in runner.get_results():
#             print(f"Task: {res.task_id}, Solved: {res.metrics['task_solved']}, "
#                   f"Test Acc: {res.metrics['test_case_accuracy']:.2f}, "
#                   f"Pixel Acc: {res.metrics['pixelwise_accuracy']:.2f}, "
#                   f"Info: {res.solver_info}")
#     except Exception as e:
#         print(f"Error during benchmark example: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         import shutil
#         if dummy_bench_task_dir.exists():
#             shutil.rmtree(dummy_bench_task_dir)
