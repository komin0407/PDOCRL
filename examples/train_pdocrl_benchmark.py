"""
PDOCRL benchmark runner for all 38 tasks.

Convenience wrapper around benchmark_runner.py specifically for PDOCRL.

Usage:
    # Full benchmark on GPU with 4 parallel processes
    python examples/train_pdocrl_benchmark.py --device cuda:0 --max_parallel 4

    # Quick test on CPU (2 tasks, 100 steps)
    python examples/train_pdocrl_benchmark.py --device cpu --quick_test

    # Specific tasks
    python examples/train_pdocrl_benchmark.py --tasks OfflineCarCircle-v0 OfflineAntRun-v0
"""

import argparse
import sys
from pathlib import Path

# Import the benchmark runner
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_runner import ALL_TASKS, run_benchmark, print_summary


# Quick test tasks (representative subset)
QUICK_TEST_TASKS = [
    "OfflineCarCircle-v0",      # BulletSafetyGym
    "OfflineCarButton1Gymnasium-v0",  # SafetyGymnasium Car
    "OfflinePointCircle1Gymnasium-v0", # SafetyGymnasium Point
    "OfflineAntVelocityGymnasium-v1",  # SafetyGymnasium Velocity
    "OfflineMetadrive-easysparse-v0",  # MetaDrive
]


def main():
    parser = argparse.ArgumentParser(
        description="Run PDOCRL benchmark on all 38 tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (e.g., 'cpu', 'cuda:0'). Default: cpu"
    )

    parser.add_argument(
        "--max_parallel",
        type=int,
        default=1,
        help="Maximum number of parallel processes. Default: 1"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Default: 0"
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific tasks to run (default: all 38 tasks)"
    )

    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run quick test on 5 representative tasks with 100 steps each"
    )

    parser.add_argument(
        "--update_steps",
        type=int,
        default=None,
        help="Number of update steps (default: use config defaults)"
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

    # Determine tasks to run
    if args.quick_test:
        tasks = QUICK_TEST_TASKS
        update_steps = 100  # Fast test
        print("\n*** QUICK TEST MODE ***")
        print(f"Running {len(tasks)} representative tasks with {update_steps} steps each\n")
    else:
        tasks = args.tasks if args.tasks else ALL_TASKS
        update_steps = args.update_steps

    if args.dry_run:
        print(f"\nWould run PDOCRL on {len(tasks)} tasks:")
        for i, task in enumerate(tasks, 1):
            print(f"  {i:2d}. {task}")
        print(f"\nDevice: {args.device}")
        print(f"Max parallel: {args.max_parallel}")
        if update_steps:
            print(f"Update steps: {update_steps}")
        return

    # Run benchmark
    results = run_benchmark(
        algorithm="pdocrl",
        tasks=tasks,
        device=args.device,
        max_parallel=args.max_parallel,
        seed=args.seed,
        update_steps=update_steps,
        extra_args=args.extra_args,
    )

    # Print summary
    print_summary(results)

    # Exit with error if any task failed
    if any(not r.success for r in results):
        exit(1)


if __name__ == "__main__":
    main()
