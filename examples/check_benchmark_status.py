"""
Check the status of benchmark runs.

This script scans the logs directory and reports on completed, running, and missing tasks.

Usage:
    python examples/check_benchmark_status.py --algorithm pdocrl
    python examples/check_benchmark_status.py --algorithm pdocrl --tasks OfflineCarCircle-v0
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


# All 38 tasks
ALL_TASKS = [
    "OfflineAntCircle-v0", "OfflineAntRun-v0", "OfflineCarCircle-v0",
    "OfflineDroneCircle-v0", "OfflineDroneRun-v0", "OfflineBallCircle-v0",
    "OfflineBallRun-v0", "OfflineCarRun-v0",
    "OfflineCarButton1Gymnasium-v0", "OfflineCarButton2Gymnasium-v0",
    "OfflineCarCircle1Gymnasium-v0", "OfflineCarCircle2Gymnasium-v0",
    "OfflineCarGoal1Gymnasium-v0", "OfflineCarGoal2Gymnasium-v0",
    "OfflineCarPush1Gymnasium-v0", "OfflineCarPush2Gymnasium-v0",
    "OfflinePointButton1Gymnasium-v0", "OfflinePointButton2Gymnasium-v0",
    "OfflinePointCircle1Gymnasium-v0", "OfflinePointCircle2Gymnasium-v0",
    "OfflinePointGoal1Gymnasium-v0", "OfflinePointGoal2Gymnasium-v0",
    "OfflinePointPush1Gymnasium-v0", "OfflinePointPush2Gymnasium-v0",
    "OfflineAntVelocityGymnasium-v1", "OfflineHalfCheetahVelocityGymnasium-v1",
    "OfflineHopperVelocityGymnasium-v1", "OfflineSwimmerVelocityGymnasium-v1",
    "OfflineWalker2dVelocityGymnasium-v1",
    "OfflineMetadrive-easysparse-v0", "OfflineMetadrive-easymean-v0",
    "OfflineMetadrive-easydense-v0", "OfflineMetadrive-mediumsparse-v0",
    "OfflineMetadrive-mediummean-v0", "OfflineMetadrive-mediumdense-v0",
    "OfflineMetadrive-hardsparse-v0", "OfflineMetadrive-hardmean-v0",
    "OfflineMetadrive-harddense-v0",
]


def check_task_status(
    log_dir: Path,
    algorithm: str,
    task: str,
    cost_limit: int = 10,
) -> Dict[str, any]:
    """
    Check the status of a task.

    Returns:
        Dictionary with status information
    """
    group = f"{task}-cost-{cost_limit}"
    group_dir = log_dir / group

    prefix = algorithm.upper()

    result = {
        'task': task,
        'group_exists': group_dir.exists(),
        'runs': [],
        'completed': False,
        'has_checkpoint': False,
        'has_best_checkpoint': False,
    }

    if not group_dir.exists():
        return result

    # Find runs for this algorithm
    for run_dir in group_dir.iterdir():
        if not run_dir.is_dir():
            continue

        if not run_dir.name.startswith(prefix):
            continue

        # Check for progress file
        progress_files = list(run_dir.rglob("progress.txt"))
        checkpoints = list(run_dir.rglob("checkpoint.pt"))
        best_checkpoints = list(run_dir.rglob("checkpoint_best.pt"))

        run_info = {
            'name': run_dir.name,
            'has_progress': len(progress_files) > 0,
            'has_checkpoint': len(checkpoints) > 0,
            'has_best_checkpoint': len(best_checkpoints) > 0,
        }

        # Parse final metrics if progress file exists
        if progress_files:
            try:
                with open(progress_files[0], 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        header = lines[0].strip().split('\t')
                        last_line = lines[-1].strip().split('\t')

                        if 'Steps' in header:
                            steps_idx = header.index('Steps')
                            steps = float(last_line[steps_idx])
                            run_info['steps'] = steps

                        if 'eval/Reward' in header:
                            reward_idx = header.index('eval/Reward')
                            reward = float(last_line[reward_idx])
                            run_info['reward'] = reward

                        if 'eval/Cost' in header:
                            cost_idx = header.index('eval/Cost')
                            cost = float(last_line[cost_idx])
                            run_info['cost'] = cost

                        run_info['completed'] = True
                        result['completed'] = True

            except Exception:
                pass

        result['runs'].append(run_info)
        result['has_checkpoint'] = result['has_checkpoint'] or run_info['has_checkpoint']
        result['has_best_checkpoint'] = result['has_best_checkpoint'] or run_info['has_best_checkpoint']

    return result


def print_status_report(
    algorithm: str,
    tasks: List[str],
    statuses: List[Dict],
) -> None:
    """Print a formatted status report."""

    completed = [s for s in statuses if s['completed']]
    missing = [s for s in statuses if not s['group_exists']]
    in_progress = [s for s in statuses if s['group_exists'] and not s['completed']]

    print(f"\n{'='*80}")
    print(f"{algorithm.upper()} Benchmark Status")
    print(f"{'='*80}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Completed: {len(completed)}")
    print(f"In progress: {len(in_progress)}")
    print(f"Not started: {len(missing)}")
    print(f"{'='*80}\n")

    if completed:
        print(f"Completed tasks ({len(completed)}):")
        print("-" * 80)
        print(f"{'Task':<45} {'Reward':>10} {'Cost':>10} {'Steps':>12}")
        print("-" * 80)

        for s in completed:
            run = s['runs'][0]  # Take first run
            task = s['task']
            reward = run.get('reward', float('nan'))
            cost = run.get('cost', float('nan'))
            steps = run.get('steps', float('nan'))

            print(f"{task:<45} {reward:>10.1f} {cost:>10.1f} {steps:>12.0f}")

        print()

    if in_progress:
        print(f"In progress ({len(in_progress)}):")
        for s in in_progress:
            print(f"  - {s['task']}")
        print()

    if missing:
        print(f"Not started ({len(missing)}):")
        for s in missing:
            print(f"  - {s['task']}")
        print()

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Check benchmark status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        help="Algorithm to check (e.g., pdocrl, bcql)"
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific tasks to check (default: all 38 tasks)"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Base log directory (default: logs)"
    )

    parser.add_argument(
        "--cost_limit",
        type=int,
        default=10,
        help="Cost limit (default: 10)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each task"
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        return

    tasks = args.tasks if args.tasks else ALL_TASKS

    # Check status of all tasks
    statuses = []
    for task in tasks:
        status = check_task_status(log_dir, args.algorithm, task, args.cost_limit)
        statuses.append(status)

        if args.verbose and status['runs']:
            print(f"\n{task}:")
            for run in status['runs']:
                print(f"  Run: {run['name']}")
                print(f"    Completed: {run.get('completed', False)}")
                if 'steps' in run:
                    print(f"    Steps: {run['steps']:.0f}")
                if 'reward' in run:
                    print(f"    Reward: {run['reward']:.2f}")
                if 'cost' in run:
                    print(f"    Cost: {run['cost']:.2f}")

    # Print summary report
    if not args.verbose:
        print_status_report(args.algorithm, tasks, statuses)


if __name__ == "__main__":
    main()
