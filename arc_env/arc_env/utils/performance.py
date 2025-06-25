import time
import functools
from typing import Callable, Any, Optional, List
from collections import defaultdict
import statistics # For mean, median, stdev

# Utilities for monitoring performance, such as timing code execution.

# --- Simple Function Timer Decorator ---
def time_function(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator that measures the execution time of a function and prints it.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter() # More precise than time.time() for short durations
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        print(f"Function '{func.__name__}' executed in {elapsed_time_ms:.3f} ms")
        return result
    return wrapper


# --- Context Manager for Timing Code Blocks ---
class CodeTimer:
    """
    A context manager to time a block of code.

    Usage:
        with CodeTimer(name="My critical section", verbose=True) as timer:
            # Code to time
            time.sleep(0.1)
        # timer.elapsed_ms will have the time taken

        # Can also be used without 'with' for manual start/stop:
        # t = CodeTimer().start()
        # ... code ...
        # t.stop()
        # print(t.elapsed_ms)
    """
    def __init__(self, name: Optional[str] = None, logger: Optional[Callable[[str], None]] = None, verbose: bool = True):
        self.name = name if name else "Unnamed code block"
        self.logger_fn = logger if logger else print # Use provided logger or print
        self.verbose = verbose
        self._start_time: Optional[float] = None
        self.elapsed_ms: Optional[float] = None

    def start(self) -> 'CodeTimer':
        """Starts the timer."""
        self._start_time = time.perf_counter()
        self.elapsed_ms = None # Reset elapsed time
        return self

    def stop(self) -> Optional[float]:
        """Stops the timer and calculates elapsed time in milliseconds."""
        if self._start_time is None:
            self.logger_fn(f"Warning: Timer '{self.name}' was stopped without being started.")
            return None

        end_time = time.perf_counter()
        self.elapsed_ms = (end_time - self._start_time) * 1000
        self._start_time = None # Reset start time to prevent re-stopping

        if self.verbose:
            self.logger_fn(f"Timer '{self.name}' stopped. Elapsed: {self.elapsed_ms:.3f} ms")
        return self.elapsed_ms

    def __enter__(self) -> 'CodeTimer':
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
        # If __exit__ returns False (or None implicitly), any exception that occurred
        # inside the 'with' block is re-raised. This is standard behavior.


# --- Profiler for Multiple Calls (Simple) ---
class CallProfiler:
    """
    A simple profiler to collect execution times for multiple calls to functions
    or named code sections and provide basic statistics.
    """
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list) # Store times in ms
        self._current_section_name: Optional[str] = None
        self._current_section_start_time: Optional[float] = None

    def start_section(self, name: str) -> None:
        """Starts timing for a named section. Nested sections are not supported by this simple version."""
        if self._current_section_name is not None:
            # Simple profiler doesn't handle nested calls well without a stack.
            # For now, just warn or overwrite. Let's warn and ignore new start.
            print(f"CallProfiler Warning: Section '{self._current_section_name}' is already active. "
                  f"Ignoring start_section for '{name}'.")
            return
        self._current_section_name = name
        self._current_section_start_time = time.perf_counter()

    def end_section(self, name: Optional[str] = None) -> None:
        """Ends timing for the currently active section (or specified name)."""
        if self._current_section_name is None:
            print("CallProfiler Warning: end_section called with no active section.")
            return
        if name is not None and name != self._current_section_name:
            print(f"CallProfiler Warning: end_section called for '{name}' but current section is '{self._current_section_name}'. Ignoring.")
            return

        end_time = time.perf_counter()
        if self._current_section_start_time is not None: # Should always be true if _current_section_name is set
            elapsed_ms = (end_time - self._current_section_start_time) * 1000
            self.timings[self._current_section_name].append(elapsed_ms)

        self._current_section_name = None
        self._current_section_start_time = None

    def record_time(self, name: str, duration_ms: float) -> None:
        """Directly records a pre-measured duration for a named event."""
        self.timings[name].append(duration_ms)

    def get_stats(self, name: Optional[str] = None) -> Union[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Calculates statistics for recorded timings.

        Args:
            name: If specified, returns stats only for this named section.
                  Otherwise, returns stats for all recorded sections.

        Returns:
            A dictionary of statistics (count, total_ms, mean_ms, median_ms, stdev_ms, min_ms, max_ms).
        """
        stats_results = {}
        sections_to_stat = [name] if name and name in self.timings else self.timings.keys()

        for section_name in sections_to_stat:
            section_times = self.timings[section_name]
            if not section_times:
                stats_results[section_name] = {"count": 0}
                continue

            stats_results[section_name] = {
                "count": len(section_times),
                "total_ms": sum(section_times),
                "mean_ms": statistics.mean(section_times),
                "median_ms": statistics.median(section_times),
                "stdev_ms": statistics.stdev(section_times) if len(section_times) > 1 else 0.0,
                "min_ms": min(section_times),
                "max_ms": max(section_times),
            }
            # Round floats for cleaner display
            for key in ["total_ms", "mean_ms", "median_ms", "stdev_ms", "min_ms", "max_ms"]:
                 if key in stats_results[section_name]:
                    stats_results[section_name][key] = round(stats_results[section_name][key], 3)

        return stats_results[name] if name and name in stats_results else stats_results


    def summary(self) -> None:
        """Prints a summary of all collected timing statistics."""
        print("\n--- CallProfiler Summary ---")
        stats = self.get_stats()
        if not stats:
            print("No timings recorded.")
            return
        for name, stat_values in stats.items():
            print(f"Section: {name}")
            if stat_values["count"] == 0:
                print("  No calls recorded.")
                continue
            print(f"  Calls: {stat_values['count']}")
            print(f"  Total: {stat_values['total_ms']} ms")
            print(f"  Mean:  {stat_values['mean_ms']} ms")
            print(f"  Median:{stat_values['median_ms']} ms")
            print(f"  StDev: {stat_values['stdev_ms']} ms")
            print(f"  Min:   {stat_values['min_ms']} ms, Max: {stat_values['max_ms']} ms")
        print("--------------------------\n")


# Example Usage:
if __name__ == "__main__":

    @time_function
    def example_func_to_time(duration_s: float):
        time.sleep(duration_s)
        return f"Slept for {duration_s}s"

    print("--- Testing @time_function decorator ---")
    example_func_to_time(0.05)
    example_func_to_time(0.12)

    print("\n--- Testing CodeTimer context manager ---")
    with CodeTimer(name="My Loop Block") as main_block_timer:
        for i in range(3):
            with CodeTimer(name=f"Inner action {i+1}", verbose=False) as inner_timer: # Less verbose for inner
                time.sleep(0.02 * (i + 1))
            # If we had a logger: main_block_timer.logger_fn(f"Inner action {i+1} took {inner_timer.elapsed_ms:.3f} ms")
            print(f"  (Log from outside) Inner action {i+1} took {inner_timer.elapsed_ms:.3f} ms")
    # main_block_timer.elapsed_ms is now available

    print("\n--- Testing CallProfiler ---")
    profiler = CallProfiler()

    # Simulate multiple calls to different "operations"
    for _ in range(5):
        profiler.start_section("data_loading")
        time.sleep(random.uniform(0.005, 0.015)) # Simulate work
        profiler.end_section() # Ends "data_loading"

    for _ in range(10):
        profiler.start_section("model_inference")
        time.sleep(random.uniform(0.020, 0.050))
        profiler.end_section("model_inference") # Explicitly naming is also fine

    # Record a pre-measured time
    profiler.record_time("external_api_call", 123.456)
    profiler.record_time("external_api_call", 150.0)

    profiler.summary()

    # Get stats for a specific section
    inference_stats = profiler.get_stats("model_inference")
    if isinstance(inference_stats, dict) and "mean_ms" in inference_stats: # Check if it's single section stats
         print(f"Model inference mean time: {inference_stats['mean_ms']:.3f} ms")

    print("\nPerformance utilities tests finished.")
