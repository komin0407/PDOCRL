"""
Benchmark runner for OSRL algorithms.

This script runs a specified algorithm on all 38 tasks in parallel.
Supports both CPU and GPU execution with configurable parallelism.

Usage:
    python examples/benchmark_runner.py --algorithm pdocrl --device cuda:0 --max_parallel 4
    python examples/benchmark_runner.py --algorithm bcql --device cpu --max_parallel 15
    python examples/benchmark_runner.py --algorithm pdocrl --tasks OfflineCarCircle-v0 OfflineAntRun-v0
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# All 38 tasks in OSRL benchmark
ALL_TASKS = [
    # bullet safety gym envs (8 tasks)
    "OfflineAntCircle-v0",
    "OfflineAntRun-v0",
    "OfflineCarCircle-v0",
    "OfflineDroneCircle-v0",
    "OfflineDroneRun-v0",
    "OfflineBallCircle-v0",
    "OfflineBallRun-v0",
    "OfflineCarRun-v0",
    # safety gymnasium: car (8 tasks)
    "OfflineCarButton1Gymnasium-v0",
    "OfflineCarButton2Gymnasium-v0",
    "OfflineCarCircle1Gymnasium-v0",
    "OfflineCarCircle2Gymnasium-v0",
    "OfflineCarGoal1Gymnasium-v0",
    "OfflineCarGoal2Gymnasium-v0",
    "OfflineCarPush1Gymnasium-v0",
    "OfflineCarPush2Gymnasium-v0",
    # safety gymnasium: point (8 tasks)
    "OfflinePointButton1Gymnasium-v0",
    "OfflinePointButton2Gymnasium-v0",
    "OfflinePointCircle1Gymnasium-v0",
    "OfflinePointCircle2Gymnasium-v0",
    "OfflinePointGoal1Gymnasium-v0",
    "OfflinePointGoal2Gymnasium-v0",
    "OfflinePointPush1Gymnasium-v0",
    "OfflinePointPush2Gymnasium-v0",
    # safety gymnasium: velocity (5 tasks)
    "OfflineAntVelocityGymnasium-v1",
    "OfflineHalfCheetahVelocityGymnasium-v1",
    "OfflineHopperVelocityGymnasium-v1",
    "OfflineSwimmerVelocityGymnasium-v1",
    "OfflineWalker2dVelocityGymnasium-v1",
    # metadrive envs (9 tasks)
    "OfflineMetadrive-easysparse-v0",
    "OfflineMetadrive-easymean-v0",
    "OfflineMetadrive-easydense-v0",
    "OfflineMetadrive-mediumsparse-v0",
    "OfflineMetadrive-mediummean-v0",
    "OfflineMetadrive-mediumdense-v0",
    "OfflineMetadrive-hardsparse-v0",
    "OfflineMetadrive-hardmean-v0",
    "OfflineMetadrive-harddense-v0",
]


@dataclass
class TaskResult:
    """Result of a single task training."""
    task: str
    success: bool
    error_msg: Optional[str] = None
    runtime_seconds: Optional[float] = None


def run_single_task(
    algorithm: str,
    task: str,
    device: str,
    seed: int = 0,
    update_steps: Optional[int] = None,
    extra_args: str = "",
) -> TaskResult:
    """
    Run training for a single task.

    Args:
        algorithm: Algorithm name (e.g., 'pdocrl', 'bcql')
        task: Task name
        device: Device to use (e.g., 'cpu', 'cuda:0')
        seed: Random seed
        update_steps: Number of update steps (None uses default from config)
        extra_args: Additional CLI arguments as string

    Returns:
        TaskResult with success status and error info
    """
    script_path = Path(__file__).parent / "train" / f"train_{algorithm}.py"
    module_name = f"examples.train.train_{algorithm}"

    if not script_path.exists():
        return TaskResult(
            task=task,
            success=False,
            error_msg=f"Training script not found: {script_path}"
        )

    cmd = [
        sys.executable,
        "-m",
        module_name,
        "--task", task,
        "--device", device,
        "--seed", str(seed),
    ]

    if update_steps is not None:
        cmd.extend(["--update_steps", str(update_steps)])

    if extra_args:
        cmd.extend(extra_args.split())

    print(f"[{task}] Starting on {device}...")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        runtime = time.time() - start_time
        print(f"[{task}] Completed in {runtime:.1f}s")
        return TaskResult(task=task, success=True, runtime_seconds=runtime)

    except subprocess.CalledProcessError as e:
        runtime = time.time() - start_time
        error_msg = f"Exit code {e.returncode}"

        # Check for NaN/Inf in output (fail-fast requirement)
        if "nan" in e.stdout.lower() or "inf" in e.stdout.lower():
            error_msg += " [NaN/Inf detected]"

        # Extract last few lines of error
        stderr_lines = e.stderr.strip().split("\n")
        if stderr_lines:
            error_msg += f": {stderr_lines[-1]}"

        print(f"[{task}] Failed after {runtime:.1f}s: {error_msg}")
        return TaskResult(
            task=task,
            success=False,
            error_msg=error_msg,
            runtime_seconds=runtime
        )

    except Exception as e:
        runtime = time.time() - start_time
        print(f"[{task}] Failed after {runtime:.1f}s: {str(e)}")
        return TaskResult(
            task=task,
            success=False,
            error_msg=str(e),
            runtime_seconds=runtime
        )


def run_benchmark(
    algorithm: str,
    tasks: List[str],
    device: str = "cpu",
    max_parallel: int = 1,
    seed: int = 0,
    update_steps: Optional[int] = None,
    extra_args: str = "",
) -> List[TaskResult]:
    """
    Run benchmark on multiple tasks in parallel.

    Args:
        algorithm: Algorithm name
        tasks: List of task names to run
        device: Device to use
        max_parallel: Maximum number of parallel processes
        seed: Random seed
        update_steps: Number of update steps (None uses default)
        extra_args: Additional CLI arguments

    Returns:
        List of TaskResult objects
    """
    print(f"\n{'='*80}")
    print(f"Running {algorithm.upper()} benchmark")
    print(f"Tasks: {len(tasks)}")
    print(f"Device: {device}")
    print(f"Max parallel: {max_parallel}")
    print(f"Seed: {seed}")
    if update_steps:
        print(f"Update steps: {update_steps}")
    print(f"{'='*80}\n")

    results = []

    if max_parallel == 1:
        # Sequential execution
        for task in tasks:
            result = run_single_task(
                algorithm, task, device, seed, update_steps, extra_args
            )
            results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(
                    run_single_task,
                    algorithm,
                    task,
                    device,
                    seed,
                    update_steps,
                    extra_args
                ): task
                for task in tasks
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

    return results


def print_summary(results: List[TaskResult]) -> None:
    """Print summary of benchmark results."""
    success_count = sum(1 for r in results if r.success)
    total = len(results)

    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks: {total}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total - success_count}")

    if success_count > 0:
        successful = [r for r in results if r.success]
        total_time = sum(r.runtime_seconds for r in successful)
        avg_time = total_time / success_count
        print(f"Total runtime: {total_time:.1f}s")
        print(f"Average runtime: {avg_time:.1f}s")

    # Print failures
    failures = [r for r in results if not r.success]
    if failures:
        print(f"\nFailed tasks:")
        for r in failures:
            print(f"  - {r.task}: {r.error_msg}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run OSRL benchmark on all tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["bc", "bcql", "bearl", "cdt", "coptidice", "cpq", "pdocrl"],
        help="Algorithm to benchmark"
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific tasks to run (default: all 38 tasks)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (e.g., 'cpu', 'cuda:0')"
    )

    parser.add_argument(
        "--max_parallel",
        type=int,
        default=1,
        help="Maximum number of parallel processes"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )

    parser.add_argument(
        "--update_steps",
        type=int,
        default=None,
        help="Number of update steps (overrides config default)"
    )

    parser.add_argument(
        "--extra_args",
        type=str,
        default="",
        help="Additional arguments to pass to training script"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print tasks that would be run without executing"
    )

    args = parser.parse_args()

    tasks = args.tasks if args.tasks else ALL_TASKS

    if args.dry_run:
        print(f"\nWould run {args.algorithm} on {len(tasks)} tasks:")
        for i, task in enumerate(tasks, 1):
            print(f"  {i:2d}. {task}")
        print(f"\nDevice: {args.device}")
        print(f"Max parallel: {args.max_parallel}")
        return

    # Run benchmark
    results = run_benchmark(
        algorithm=args.algorithm,
        tasks=tasks,
        device=args.device,
        max_parallel=args.max_parallel,
        seed=args.seed,
        update_steps=args.update_steps,
        extra_args=args.extra_args,
    )

    # Print summary
    print_summary(results)

    # Exit with error if any task failed
    if any(not r.success for r in results):
        exit(1)


if __name__ == "__main__":
    main()
