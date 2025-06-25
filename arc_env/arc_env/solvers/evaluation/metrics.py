from __future__ import annotations

from typing import List, Dict, Any, Optional, Union # Union not used here but fine
import numpy as np

from arc_env.data.loaders.base import ARCTaskData # Protocol
# from arc_env.solvers.base.base_solver import BaseSolver # Not directly needed for these metric functions
from arc_env.solvers.base.solver_utils import compare_grids, solution_accuracy

# This module defines metrics for evaluating solver performance on ARC tasks.

def task_solved_metric(
    predicted_outputs: List[np.ndarray],
    target_outputs: List[np.ndarray]
) -> float:
    """
    Determines if a task is solved. A task is solved if all its test cases
    are correctly predicted.

    Args:
        predicted_outputs: A list of grids predicted by the solver for each test case.
        target_outputs: A list of the ground truth solution grids for each test case.

    Returns:
        1.0 if all test cases are solved correctly, 0.0 otherwise.
        Returns 0.0 if lists are empty or have mismatched lengths.
    """
    if not predicted_outputs or not target_outputs:
        return 0.0 # Cannot determine if empty
    if len(predicted_outputs) != len(target_outputs):
        print(f"Warning: Mismatched number of predicted ({len(predicted_outputs)}) "
              f"and target ({len(target_outputs)}) outputs for task_solved_metric.")
        return 0.0

    for pred_grid, target_grid in zip(predicted_outputs, target_outputs):
        if not compare_grids(pred_grid, target_grid):
            return 0.0 # At least one test case failed

    return 1.0 # All test cases solved


def test_case_accuracy_metric(
    predicted_outputs: List[np.ndarray],
    target_outputs: List[np.ndarray]
) -> float:
    """
    Calculates the average accuracy across all test cases for a single task.
    Accuracy for each test case is 1.0 if perfectly solved, 0.0 otherwise.
    (This is equivalent to proportion of test cases solved).

    Args:
        predicted_outputs: List of predicted grids.
        target_outputs: List of ground truth solution grids.

    Returns:
        The average accuracy (proportion of test cases solved).
        Returns 0.0 if lists are empty or mismatched.
    """
    if not predicted_outputs or not target_outputs:
        return 0.0
    if len(predicted_outputs) != len(target_outputs):
        print(f"Warning: Mismatched number of predicted ({len(predicted_outputs)}) "
              f"and target ({len(target_outputs)}) outputs for test_case_accuracy_metric.")
        return 0.0

    num_cases = len(target_outputs)
    correct_cases = 0
    for pred_grid, target_grid in zip(predicted_outputs, target_outputs):
        if compare_grids(pred_grid, target_grid):
            correct_cases += 1

    return float(correct_cases) / num_cases if num_cases > 0 else 0.0


def pixelwise_accuracy_metric(
    predicted_outputs: List[np.ndarray],
    target_outputs: List[np.ndarray]
) -> float:
    """
    Calculates the average pixel-wise accuracy across all test cases for a task.
    For each test case, pixel-wise accuracy is computed. Then these scores are averaged.

    Args:
        predicted_outputs: List of predicted grids.
        target_outputs: List of ground truth solution grids.

    Returns:
        The average pixel-wise accuracy.
        Returns 0.0 if lists are empty or mismatched.
    """
    if not predicted_outputs or not target_outputs:
        return 0.0
    if len(predicted_outputs) != len(target_outputs):
        print(f"Warning: Mismatched number of predicted ({len(predicted_outputs)}) "
              f"and target ({len(target_outputs)}) outputs for pixelwise_accuracy_metric.")
        return 0.0

    num_cases = len(target_outputs)
    total_pixel_accuracy_sum = 0.0
    for pred_grid, target_grid in zip(predicted_outputs, target_outputs):
        total_pixel_accuracy_sum += solution_accuracy(pred_grid, target_grid) # Uses solver_utils

    return total_pixel_accuracy_sum / num_cases if num_cases > 0 else 0.0


# Higher-level evaluation results structure
class TaskEvaluationResult:
    def __init__(self, task_id: str):
        self.task_id: str = task_id
        self.metrics: Dict[str, Any] = {} # Store various metric scores
        self.predicted_test_outputs: List[np.ndarray] = []
        self.target_test_outputs: List[np.ndarray] = []
        self.solver_info: Dict[str, Any] = {} # E.g., steps taken, time, errors

    def add_metric(self, name: str, value: Any):
        self.metrics[name] = value

    def __repr__(self) -> str:
        return f"TaskEvaluationResult(task_id='{self.task_id}', metrics={self.metrics}, solver_info={self.solver_info})"


def evaluate_single_task_attempt(
    task_id: str,
    predicted_grids: List[np.ndarray], # Solver's output grids for each test case in the task
    target_grids: List[np.ndarray],   # Ground truth output grids for each test case
    solver_run_info: Optional[Dict[str, Any]] = None # Info like time taken, steps, etc.
) -> TaskEvaluationResult:
    """
    Evaluates a solver's attempt on a single ARC task and computes standard metrics.
    """
    result = TaskEvaluationResult(task_id)
    result.predicted_test_outputs = [np.copy(p) for p in predicted_grids] # Store copies
    result.target_test_outputs = [np.copy(t) for t in target_grids]
    if solver_run_info:
        result.solver_info = solver_run_info

    result.add_metric("task_solved", task_solved_metric(predicted_grids, target_grids))
    result.add_metric("test_case_accuracy", test_case_accuracy_metric(predicted_grids, target_grids))
    result.add_metric("pixelwise_accuracy", pixelwise_accuracy_metric(predicted_grids, target_grids))

    # Could add more metrics:
    # - Shape match accuracy (if predicted shape must match target shape)
    # - Number of colors match accuracy
    # - Structural similarity index (if applicable to discrete grids)

    return result


# Example Usage
if __name__ == '__main__':
    # Example test cases for a task
    # Task 1: 2 test cases
    pred_task1_case1 = np.array([[1,0],[0,1]])
    target_task1_case1 = np.array([[1,0],[0,1]]) # Correct

    pred_task1_case2 = np.array([[2,2],[2,2]])
    target_task1_case2 = np.array([[2,2],[2,3]]) # Incorrect (1 pixel off)

    preds_task1 = [pred_task1_case1, pred_task1_case2]
    targets_task1 = [target_task1_case1, target_task1_case2]

    eval_res_task1 = evaluate_single_task_attempt(
        "task_example_01",
        preds_task1,
        targets_task1,
        solver_run_info={"time_ms": 1200, "steps_total": 15}
    )
    print("Evaluation for Task 1:")
    print(eval_res_task1)
    # Expected for Task 1:
    # task_solved: 0.0 (case 2 failed)
    # test_case_accuracy: 0.5 (1 out of 2 solved)
    # pixelwise_accuracy: (1.0 [case1] + 0.75 [case2 for 2x2 grid]) / 2 = 0.875
    assert eval_res_task1.metrics["task_solved"] == 0.0
    assert eval_res_task1.metrics["test_case_accuracy"] == 0.5
    assert abs(eval_res_task1.metrics["pixelwise_accuracy"] - ((1.0 + 0.75)/2.0)) < 1e-9


    # Task 2: 1 test case, solved perfectly
    pred_task2_case1 = np.array([[5]])
    target_task2_case1 = np.array([[5]])
    preds_task2 = [pred_task2_case1]
    targets_task2 = [target_task2_case1]

    eval_res_task2 = evaluate_single_task_attempt("task_example_02", preds_task2, targets_task2)
    print("\nEvaluation for Task 2:")
    print(eval_res_task2)
    # Expected for Task 2:
    # task_solved: 1.0
    # test_case_accuracy: 1.0
    # pixelwise_accuracy: 1.0
    assert eval_res_task2.metrics["task_solved"] == 1.0
    assert eval_res_task2.metrics["test_case_accuracy"] == 1.0
    assert eval_res_task2.metrics["pixelwise_accuracy"] == 1.0

    # Task 3: Mismatched number of predictions/targets
    preds_task3 = [np.array([[1]])]
    targets_task3 = [np.array([[1]]), np.array([[2]])] # Target has 2 cases
    eval_res_task3 = evaluate_single_task_attempt("task_example_03", preds_task3, targets_task3)
    print("\nEvaluation for Task 3 (mismatched lists):")
    print(eval_res_task3)
    # Expected all metrics to be 0.0 due to mismatch warning
    assert eval_res_task3.metrics["task_solved"] == 0.0
    assert eval_res_task3.metrics["test_case_accuracy"] == 0.0
    assert eval_res_task3.metrics["pixelwise_accuracy"] == 0.0

    print("\nAll metric examples passed.")
