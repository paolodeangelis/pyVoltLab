import time
from collections.abc import Callable
from datetime import datetime, timedelta
from functools import wraps
from typing import Any

import numpy as np


class MethodProfiler:
    """
    A utility class for profiling methods by tracking their execution time
    and the number of calls.

    Tracks:
        - Total execution time
        - Number of calls
        - Minimum execution time per call
        - Maximum execution time per call

    Attributes:
        name (str): A name to identify the profiler instance.
        stats (Dict[str, Dict[str, float]]): A dictionary storing profiling statistics
            for each tracked method.

    Example:
        >>> profiler = MethodProfiler("ExampleProfiler")
        >>> @profiler.track
        ... def example_function():
        ...     time.sleep(1)
        ...
        >>> example_function()
        >>> print(profiler.report())
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the MethodProfiler.

        Args:
            name (str): A name for the profiler instance.
        """
        self.name = name
        self.stats: dict[str, dict[str, float]] = {}

    def track(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        A decorator to track the execution time and call count of a function.

        Args:
            func (Callable): The function to be tracked.

        Returns:
            Callable: The wrapped function with tracking.
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            elapsed = end - start
            func_name = func.__name__

            # Initialize stats for the function if not already present
            if func_name not in self.stats:
                self.stats[func_name] = {
                    "time": 0.0,  # Total time spent
                    "calls": 0,  # Number of calls
                    "min_time": float("inf"),  # Minimum time for a single call
                    "max_time": 0.0,  # Maximum time for a single call
                }

            # Update statistics
            self.stats[func_name]["time"] += elapsed
            self.stats[func_name]["calls"] += 1
            self.stats[func_name]["min_time"] = min(self.stats[func_name]["min_time"], elapsed)
            self.stats[func_name]["max_time"] = max(self.stats[func_name]["max_time"], elapsed)

            return result

        return wrapper

    def _format_time(self, seconds: float) -> str:
        """
        Format time dynamically based on the total time.

        Args:
            seconds (float): Time in seconds.

        Returns:
            str: Formatted time string with the most appropriate unit.
        """
        if seconds < 60:
            return f"{seconds:.3f} s"
        elif seconds < 3600:
            return f"{seconds / 60:.3f} min"
        else:
            return f"{seconds / 3600:.3f} hr"

    def report(self) -> list[str]:
        """
        Generate a profiling report for all tracked methods.

        Returns:
            List[str]: A list of strings representing the profiling report.
        """
        total_time = sum(stat["time"] for stat in self.stats.values())
        report_lines = [f"-- {self.name} ".ljust(120, "-")]
        report_lines.append("")
        report_lines.append(
            f"{'Method':<30} | {'Tot. Time':<16} {'Avg. Time (ms)':<16} {'Min Time (ms)':<16} {'Max Time (ms)':<16} {'Calls':<10} {'% Time':<10}"
        )
        report_lines.append("-" * 30 + "-+-" + "-" * (120 - 33))

        # Sort stats by total time in descending order
        sorted_stats = {k: v for k, v in sorted(self.stats.items(), key=lambda item: item[1]["time"], reverse=True)}

        for method, stat in sorted_stats.items():
            percentage = (stat["time"] / total_time) * 100 if total_time else 0
            avg_time = stat["time"] / stat["calls"]
            report_lines.append(
                f"{method:<30} | {self._format_time(stat['time']):<16} "
                f"{avg_time * 1e3:<16.3f} {stat['min_time'] * 1e3:<16.3f} "
                f"{stat['max_time'] * 1e3:<16.3f} {stat['calls']:<10} {percentage:<10.2f}"
            )
        report_lines.append("-" * 30 + "---" + "-" * (120 - 33))
        return report_lines


class MCProfiler:
    """A profiler for tracking and analyzing Monte Carlo step execution times.

    This class implements Welford's online algorithm to calculate running statistics
    (mean and standard deviation) of execution times without storing individual values.
    It can be used as a decorator to automatically track method execution times.

    Attributes:
        _n (int): Number of samples collected.
        _mean (float): Running mean of execution times in seconds.
        _m2 (float): Running sum of squares of differences from mean.
        start_time (float | None): Timestamp when profiling started.

    Example:
        >>> profiler = MCProfiler()
        >>>
        >>> class MCSimulation:
        ...     @profiler.track
        ...     def mc_step(self):
        ...         # Monte Carlo step implementation
        ...         pass
        ...
        ...     def run(self, steps: int):
        ...         for i in range(steps):
        ...             self.mc_step()
        ...             if i % 1000 == 0:
        ...                 stats = profiler.get_stats(i, steps)
        ...                 print(f"Completion time: {stats['estimated_completion']}")
    """

    def __init__(self) -> None:
        """Initialize the profiler with zeroed statistics."""
        self._n: int = 0
        self._mean: float = 0.0
        self._var: float = 0.0
        self._sum: float = 0.0
        self._sum2: float = 0.0
        self.start_time: float | None = None

    def track(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to track execution time of a method.

        Args:
            func (Callable): The function to be decorated.

        Returns:
            A wrapped function that tracks execution time.

        Example:
            >>> @profiler.track
            ... def mc_step(self):
            ...     # Method implementation
            ...     pass
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Initialize start_time on first call
            if self.start_time is None:
                self.start_time = time.time()

            # Time the function execution
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start

            # Update running statistics using Welford's online algorithm
            self._n += 1
            self._sum += duration
            self._sum2 += np.square(duration)
            self._mean = self._sum / self._n
            if self._n > 1:
                self._var = (self._sum2 - (self._sum * self._sum) / self._n) / (self._n - 1)

            return result

        return wrapper

    @property
    def mean_time(self) -> float:
        """Calculate the mean execution time.

        Returns:
            Mean execution time in seconds.
        """
        return self._mean if self._n > 0 else 0.0

    @property
    def std_time(self) -> float:
        """Calculate the standard deviation of execution time.

        Returns:
            Standard deviation of execution time in seconds.
        """
        return np.sqrt(self._var) if self._n > 1 else 0.0

    @property
    def steps_per_second(self) -> float:
        """Calculate the number of steps executed per second.

        Returns:
            Steps per second rate.
        """
        return 1.0 / self.mean_time if self.mean_time > 0 else 0.0

    @property
    def steps_per_hour(self) -> float:
        """Calculate the number of steps executed per hour.

        Returns:
            Steps per hour rate.
        """
        return self.steps_per_second * 3600

    @property
    def std_step_per_hour(self) -> float:
        """Calculate the standard deviation of number of steps executed per hour.

        Returns:
            Std steps per hour rate.
        """
        return np.sqrt(np.power(self._mean, -2) * self._var) * 3600 if self._n > 1 else 0.0

    def estimate_remaining_time(self, current_step: int, total_steps: int) -> timedelta:
        """Estimate remaining time based on current progress.

        Args:
            current_step: Current step number in the simulation.
            total_steps: Total number of steps to be executed.

        Returns:
            Estimated time remaining as a timedelta object.

        Example:
            >>> remaining = profiler.estimate_remaining_time(1000, 10000)
            >>> print(f"Time remaining: {remaining}")
        """
        steps_remaining = total_steps - current_step
        estimated_seconds = steps_remaining * self.mean_time
        return timedelta(seconds=estimated_seconds)

    def estimate_completion_time(self, current_step: int, total_steps: int) -> datetime:
        """Estimate completion time based on current progress.

        Args:
            current_step: Current step number in the simulation.
            total_steps: Total number of steps to be executed.

        Returns:
            Estimated completion time as a datetime object.

        Example:
            >>> completion_time = profiler.estimate_completion_time(1000, 10000)
            >>> print(f"Expected completion: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
        """
        remaining = self.estimate_remaining_time(current_step, total_steps)
        return datetime.now() + remaining

    def get_stats(self) -> dict[str, Any]:
        """Get all relevant statistics and estimates.

        This method compiles all profiling statistics into a formatted dictionary
        for easy reporting.

        Returns:
            Dictionary containing formatted statistics with the following keys:
                - mean_step_time: Average time per step
                - std_step_time: Standard deviation of step time
                - steps_per_second: Steps executed per second
                - steps_per_hour: Steps executed per hour
                - elapsed_time: Total time elapsed

        Example:
            >>> stats = profiler.get_stats(1000, 10000)
            >>> print(f"Progress: {stats['elapsed_time']} elapsed, "
            ...       f"{stats['remaining_time']} remaining")
        """
        return {
            "mean_step_time": self.mean_time,
            "std_step_time": self.std_time,
            "steps_per_second": self.steps_per_second,
            "steps_per_hour": self.steps_per_hour,
            "elapsed_time": (
                str(timedelta(seconds=int(time.time() - self.start_time))) if self.start_time else "0:00:00"
            ),
        }
