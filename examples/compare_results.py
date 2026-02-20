"""
Compare OSRL algorithm results across tasks.

This script extracts results from local logs and optionally from W&B,
and generates comparison tables for different algorithms.

Usage:
    # Compare all algorithms on all tasks (from local logs)
    python examples/compare_results.py --algorithms pdocrl bcql bearl coptidice cpq

    # Compare specific tasks
    python examples/compare_results.py --algorithms pdocrl bcql --tasks OfflineCarCircle-v0

    # Save to CSV
    python examples/compare_results.py --algorithms pdocrl bcql --output results.csv

    # Use W&B API (requires wandb login)
    python examples/compare_results.py --algorithms pdocrl bcql --use_wandb --wandb_project OSRL-baselines
"""

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("Error: Required packages not installed.")
    print("Install with: pip install numpy pandas")
    sys.exit(1)

warnings.filterwarnings('ignore')


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


def parse_progress_file(filepath: Path) -> Optional[Dict[str, float]]:
    """
    Parse progress.txt file and extract final metrics.

    Args:
        filepath: Path to progress.txt file

    Returns:
        Dictionary with metrics or None if parsing failed
    """
    try:
        # Read the file
        with open(filepath, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            return None

        # Parse header
        header = lines[0].strip().split('\t')

        # Parse last line (final metrics)
        last_line = lines[-1].strip().split('\t')

        if len(header) != len(last_line):
            return None

        metrics = {}
        for key, value in zip(header, last_line):
            try:
                metrics[key] = float(value)
            except (ValueError, TypeError):
                pass

        # Extract relevant metrics
        result = {
            'reward': metrics.get('eval/Reward', np.nan),
            'cost': metrics.get('eval/Cost', np.nan),
            'length': metrics.get('eval/Length', np.nan),
            'steps': metrics.get('Steps', np.nan),
        }

        return result

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def find_best_run(log_dir: Path, algorithm: str, task: str, cost_limit: int = 10) -> Optional[Dict[str, float]]:
    """
    Find the best run for a given algorithm and task.

    Args:
        log_dir: Base log directory
        algorithm: Algorithm name (lowercase)
        task: Task name
        cost_limit: Cost limit used in training

    Returns:
        Best run metrics or None
    """
    # Construct group directory name
    group = f"{task}-cost-{cost_limit}"
    group_dir = log_dir / group

    if not group_dir.exists():
        return None

    # Find all runs for this algorithm
    prefix = algorithm.upper()
    runs = []

    for run_dir in group_dir.iterdir():
        if not run_dir.is_dir():
            continue

        # Check if this is a run for the target algorithm
        if not run_dir.name.startswith(prefix):
            continue

        # Look for progress.txt in subdirectory
        progress_files = list(run_dir.rglob("progress.txt"))

        for progress_file in progress_files:
            metrics = parse_progress_file(progress_file)
            if metrics is not None:
                runs.append({
                    'run_name': run_dir.name,
                    'metrics': metrics,
                })

    if not runs:
        return None

    # Select best run (lowest cost, then highest reward)
    best_run = min(
        runs,
        key=lambda x: (x['metrics']['cost'], -x['metrics']['reward'])
    )

    return best_run['metrics']


def extract_local_results(
    algorithms: List[str],
    tasks: List[str],
    log_dir: Path,
    cost_limit: int = 10,
) -> pd.DataFrame:
    """
    Extract results from local log files.

    Args:
        algorithms: List of algorithm names
        tasks: List of task names
        log_dir: Base log directory
        cost_limit: Cost limit used in training

    Returns:
        DataFrame with results
    """
    results = []

    for task in tasks:
        row = {'task': task}

        for algo in algorithms:
            metrics = find_best_run(log_dir, algo, task, cost_limit)

            if metrics is not None:
                row[f'{algo}_reward'] = metrics['reward']
                row[f'{algo}_cost'] = metrics['cost']
                row[f'{algo}_steps'] = metrics['steps']
            else:
                row[f'{algo}_reward'] = np.nan
                row[f'{algo}_cost'] = np.nan
                row[f'{algo}_steps'] = np.nan

        results.append(row)

    return pd.DataFrame(results)


def extract_wandb_results(
    algorithms: List[str],
    tasks: List[str],
    project: str,
    entity: Optional[str] = None,
    cost_limit: int = 10,
) -> pd.DataFrame:
    """
    Extract results from W&B API.

    Args:
        algorithms: List of algorithm names
        tasks: List of task names
        project: W&B project name
        entity: W&B entity (username or team)
        cost_limit: Cost limit used in training

    Returns:
        DataFrame with results
    """
    try:
        import wandb
    except ImportError:
        print("Error: wandb not installed. Install with: pip install wandb")
        return pd.DataFrame()

    api = wandb.Api()

    results = []

    for task in tasks:
        row = {'task': task}
        group = f"{task}-cost-{cost_limit}"

        for algo in algorithms:
            prefix = algo.upper()

            # Query runs
            filters = {
                "group": group,
                "state": "finished",
            }

            try:
                if entity:
                    runs = api.runs(f"{entity}/{project}", filters=filters)
                else:
                    runs = api.runs(project, filters=filters)

                # Filter for this algorithm
                algo_runs = [r for r in runs if r.name.startswith(prefix)]

                if not algo_runs:
                    row[f'{algo}_reward'] = np.nan
                    row[f'{algo}_cost'] = np.nan
                    continue

                # Find best run
                best_run = None
                best_cost = float('inf')
                best_reward = -float('inf')

                for run in algo_runs:
                    summary = run.summary

                    cost = summary.get('eval/Cost', float('inf'))
                    reward = summary.get('eval/Reward', -float('inf'))

                    if cost < best_cost or (cost == best_cost and reward > best_reward):
                        best_cost = cost
                        best_reward = reward
                        best_run = run

                if best_run:
                    row[f'{algo}_reward'] = best_reward
                    row[f'{algo}_cost'] = best_cost
                else:
                    row[f'{algo}_reward'] = np.nan
                    row[f'{algo}_cost'] = np.nan

            except Exception as e:
                print(f"Error querying W&B for {task}/{algo}: {e}")
                row[f'{algo}_reward'] = np.nan
                row[f'{algo}_cost'] = np.nan

        results.append(row)

    return pd.DataFrame(results)


def format_comparison_table(df: pd.DataFrame, algorithms: List[str]) -> str:
    """
    Format results DataFrame as a nice comparison table.

    Args:
        df: Results DataFrame
        algorithms: List of algorithm names

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("\n" + "="*120)
    lines.append("ALGORITHM COMPARISON")
    lines.append("="*120)

    # Header
    header = "Task".ljust(40)
    for algo in algorithms:
        header += f" | {algo.upper():^24}"
    lines.append(header)
    lines.append("-" * 120)

    # Subheader
    subheader = " " * 40
    for _ in algorithms:
        subheader += " |   Reward      Cost   "
    lines.append(subheader)
    lines.append("-" * 120)

    # Rows
    for _, row in df.iterrows():
        line = row['task'].ljust(40)

        for algo in algorithms:
            reward = row[f'{algo}_reward']
            cost = row[f'{algo}_cost']

            if pd.notna(reward) and pd.notna(cost):
                reward_str = f"{reward:8.1f}"
                cost_str = f"{cost:8.1f}"
            else:
                reward_str = "     N/A"
                cost_str = "     N/A"

            line += f" | {reward_str}  {cost_str}"

        lines.append(line)

    lines.append("="*120)

    # Summary statistics
    lines.append("\nSUMMARY STATISTICS")
    lines.append("-" * 120)

    summary_line = "Mean".ljust(40)
    for algo in algorithms:
        reward_col = f'{algo}_reward'
        cost_col = f'{algo}_cost'

        if reward_col in df.columns and cost_col in df.columns:
            mean_reward = df[reward_col].mean()
            mean_cost = df[cost_col].mean()

            if pd.notna(mean_reward) and pd.notna(mean_cost):
                summary_line += f" | {mean_reward:8.1f}  {mean_cost:8.1f}"
            else:
                summary_line += " |      N/A       N/A"
        else:
            summary_line += " |      N/A       N/A"

    lines.append(summary_line)
    lines.append("="*120 + "\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare OSRL algorithm results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["pdocrl", "bcql", "bearl", "coptidice", "cpq"],
        help="Algorithms to compare (default: all baselines + pdocrl)"
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Tasks to compare (default: all 38 tasks)"
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
        help="Cost limit used in training (default: 10)"
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Extract results from W&B instead of local logs"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="OSRL-baselines",
        help="W&B project name (default: OSRL-baselines)"
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (username or team)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (optional)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["table", "csv", "both"],
        default="table",
        help="Output format (default: table)"
    )

    args = parser.parse_args()

    tasks = args.tasks if args.tasks else ALL_TASKS

    print(f"Comparing {len(args.algorithms)} algorithms on {len(tasks)} tasks...")

    # Extract results
    if args.use_wandb:
        print("Extracting results from W&B...")
        df = extract_wandb_results(
            args.algorithms,
            tasks,
            args.wandb_project,
            args.wandb_entity,
            args.cost_limit,
        )
    else:
        log_dir = Path(args.log_dir)
        if not log_dir.exists():
            print(f"Error: Log directory not found: {log_dir}")
            return

        print(f"Extracting results from local logs: {log_dir}")
        df = extract_local_results(
            args.algorithms,
            tasks,
            log_dir,
            args.cost_limit,
        )

    if df.empty:
        print("No results found!")
        return

    # Output results
    if args.format in ["table", "both"]:
        table = format_comparison_table(df, args.algorithms)
        print(table)

    if args.format in ["csv", "both"] or args.output:
        output_path = args.output if args.output else "osrl_comparison.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
