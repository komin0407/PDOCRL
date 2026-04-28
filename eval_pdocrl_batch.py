"""
Batch evaluation of PDOCRL models for seeds 0, 5, 10 and cost thresholds 20, 40, 80.
Computes normalized reward and cost, reports mean ± std per task.
"""
import os
import sys
import glob
import yaml
import numpy as np
import types
from collections import defaultdict

import bullet_safety_gym  # noqa
import gymnasium as gym  # noqa
import torch
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from osrl.algorithms import PDOCRL, PDOCRLTrainer
from osrl.common.exp_util import load_config_and_model, seed_all

TASKS = [
    "OfflineBallRun-v0",
    "OfflineCarRun-v0",
    "OfflineDroneRun-v0",
    "OfflineAntRun-v0",
    "OfflineBallCircle-v0",
    "OfflineCarCircle-v0",
    "OfflineDroneCircle-v0",
    "OfflineAntCircle-v0",
]

TARGET_SEEDS = {0, 5, 10}
TARGET_COSTS = {20, 40, 80}
EVAL_EPISODES = 20
LOGS_DIR = "/Users/komin0407/OSRL/logs"
DEVICE = "cpu"
THREADS = 4


def find_pdocrl_runs():
    """Find all PDOCRL (not PDOCRL-W/-W) experiment dirs and index by (task, seed, cost)."""
    runs = defaultdict(list)  # key: (task, seed, cost) -> list of exp_dirs

    for task in TASKS:
        task_dirs = glob.glob(os.path.join(LOGS_DIR, f"{task}-cost-*"))
        for cost_dir in task_dirs:
            # Extract cost from directory name
            cost_str = cost_dir.rsplit("-cost-", 1)[-1]
            try:
                cost = int(cost_str)
            except ValueError:
                continue
            if cost not in TARGET_COSTS:
                continue

            # Look for PDOCRL runs (not PDOCRL-W)
            for run_dir in os.listdir(cost_dir):
                if "PDOCRL-W" in run_dir or "PDOCRL_W" in run_dir or "PDOCRL_w" in run_dir:
                    continue
                if not run_dir.startswith("PDOCRL_"):
                    continue

                exp_dir = os.path.join(cost_dir, run_dir, run_dir)
                if not os.path.isdir(exp_dir):
                    # Try without the repeated subdir
                    exp_dir = os.path.join(cost_dir, run_dir)

                config_path = os.path.join(exp_dir, "config.yaml")
                model_path = os.path.join(exp_dir, "checkpoint", "model.pt")

                if not os.path.exists(config_path) or not os.path.exists(model_path):
                    continue

                with open(config_path) as f:
                    cfg = yaml.safe_load(f)

                seed = cfg.get("seed", 0)
                if seed not in TARGET_SEEDS:
                    continue

                runs[(task, seed, cost)].append(exp_dir)

    return runs


def pick_best_run(exp_dirs):
    """Among multiple runs for same (task, seed, cost), prefer initlam1.0 over fixthresh."""
    if len(exp_dirs) == 1:
        return exp_dirs[0]

    preferred = [d for d in exp_dirs if "initlam1.0" in d and "fixthresh" not in d and "eplen500" not in d]
    if preferred:
        return preferred[-1]  # Take most recent
    return exp_dirs[-1]


def evaluate_model(exp_dir):
    """Run PDOCRL evaluation and return (raw_reward, raw_cost)."""
    cfg, model = load_config_and_model(exp_dir)
    seed_all(cfg["seed"])
    torch.set_num_threads(THREADS)

    env = wrap_env(
        env=gym.make(cfg["task"]),
        reward_scale=cfg["reward_scale"],
    )
    env = OfflineEnvWrapper(env)
    env.set_target_cost(cfg["cost_limit"])

    pdocrl_model = PDOCRL(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        a_hidden_sizes=cfg["a_hidden_sizes"],
        c_hidden_sizes=cfg["c_hidden_sizes"],
        w_hidden_sizes=cfg["w_hidden_sizes"],
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        num_q=cfg["num_q"],
        slater_phi=cfg["slater_phi"],
        cost_limit=cfg["cost_limit"],
        episode_len=cfg["episode_len"],
        device=DEVICE,
    )
    pdocrl_model.load_state_dict(model["model_state"])
    pdocrl_model.to(DEVICE)

    trainer = PDOCRLTrainer(
        pdocrl_model,
        env,
        reward_scale=cfg["reward_scale"],
        cost_scale=cfg["cost_scale"],
        device=DEVICE,
    )

    ret, cost, length = trainer.evaluate(EVAL_EPISODES)
    norm_ret, norm_cost = env.get_normalized_score(ret, cost)
    return norm_ret, norm_cost, ret, cost, cfg["cost_limit"]


def main():
    torch.set_num_threads(THREADS)
    runs = find_pdocrl_runs()

    # Check coverage
    print("=" * 70)
    print("Run coverage:")
    missing = []
    for task in TASKS:
        for seed in sorted(TARGET_SEEDS):
            for cost in sorted(TARGET_COSTS):
                key = (task, seed, cost)
                if key not in runs:
                    missing.append(key)
                    print(f"  MISSING: {task} seed={seed} cost={cost}")
                else:
                    chosen = pick_best_run(runs[key])
                    print(f"  OK: {task} seed={seed} cost={cost} -> {os.path.basename(chosen)}")
    print("=" * 70)

    if missing:
        print(f"\nWarning: {len(missing)} runs missing. Proceeding with available runs.\n")

    # Run evaluations
    results = defaultdict(lambda: {"norm_rewards": [], "norm_costs": []})

    total_runs = sum(len(v) > 0 for v in [runs.get((t, s, c)) for t in TASKS for s in TARGET_SEEDS for c in TARGET_COSTS] if v is not None)
    done = 0

    for task in TASKS:
        for seed in sorted(TARGET_SEEDS):
            for cost in sorted(TARGET_COSTS):
                key = (task, seed, cost)
                if key not in runs:
                    continue
                exp_dir = pick_best_run(runs[key])
                done += 1
                print(f"\n[{done}] Evaluating {task} seed={seed} cost={cost}")
                print(f"  Path: {exp_dir}")
                try:
                    norm_ret, norm_cost, raw_ret, raw_cost, cost_limit = evaluate_model(exp_dir)
                    results[task]["norm_rewards"].append(norm_ret)
                    results[task]["norm_costs"].append(norm_cost)
                    print(f"  raw reward={raw_ret:.2f}, raw cost={raw_cost:.2f}")
                    print(f"  norm reward={norm_ret:.4f}, norm cost={norm_cost:.4f}")
                except Exception as e:
                    print(f"  ERROR: {e}")

    # Print results table
    print("\n" + "=" * 80)
    print(f"{'Task':<22} {'Norm Reward':>20} {'Norm Cost':>20}")
    print("=" * 80)

    for task in TASKS:
        short = task.replace("Offline", "").replace("-v0", "")
        nr = results[task]["norm_rewards"]
        nc = results[task]["norm_costs"]
        if nr:
            rstr = f"{np.mean(nr):.4f} ± {np.std(nr):.4f}"
        else:
            rstr = "N/A"
        if nc:
            cstr = f"{np.mean(nc):.4f} ± {np.std(nc):.4f}"
        else:
            cstr = "N/A"
        print(f"{short:<22} {rstr:>20} {cstr:>20}")

    print("=" * 80)

    # Save raw results
    import json
    out = {}
    for task in TASKS:
        out[task] = {
            "norm_rewards": results[task]["norm_rewards"],
            "norm_costs": results[task]["norm_costs"],
        }
    with open("/tmp/pdocrl_eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nRaw results saved to /tmp/pdocrl_eval_results.json")


if __name__ == "__main__":
    main()
