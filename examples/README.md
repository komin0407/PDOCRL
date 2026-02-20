# OSRL Examples

This directory contains training scripts, evaluation scripts, and benchmark runners for all OSRL algorithms.

## Quick Links

### Run PDOCRL Benchmark
```bash
# Quick test (5 tasks, 5 min)
python train_pdocrl_benchmark.py --quick_test --device cpu

# Full benchmark (38 tasks, 6-8 hours)
python train_pdocrl_benchmark.py --device cpu --max_parallel 15
```

See [QUICKSTART_BENCHMARK.md](../QUICKSTART_BENCHMARK.md) for more commands.

### Run Any Algorithm Benchmark
```bash
# Example: Run BCQL on all tasks
python benchmark_runner.py --algorithm bcql --device cpu --max_parallel 15
```

### Compare Results
```bash
# Compare algorithms
python compare_results.py --algorithms pdocrl bcql bearl coptidice cpq

# Export to CSV
python compare_results.py --algorithms pdocrl bcql --output results.csv
```

### Check Progress
```bash
python check_benchmark_status.py --algorithm pdocrl
```

## Directory Structure

```
examples/
├── train/                          # Training scripts
│   ├── train_bc.py
│   ├── train_bcql.py
│   ├── train_bearl.py
│   ├── train_cdt.py
│   ├── train_coptidice.py
│   ├── train_cpq.py
│   └── train_pdocrl.py
├── eval/                           # Evaluation scripts
│   ├── eval_bc.py
│   ├── eval_bcql.py
│   ├── eval_bearl.py
│   ├── eval_cdt.py
│   ├── eval_coptidice.py
│   ├── eval_cpq.py
│   └── eval_pdocrl.py
├── configs/                        # Algorithm configs
│   ├── bc_configs.py
│   ├── bcql_configs.py
│   ├── bearl_configs.py
│   ├── cdt_configs.py
│   ├── coptidice_configs.py
│   ├── cpq_configs.py
│   └── pdocrl_configs.py
├── benchmark_runner.py             # General benchmark runner
├── train_pdocrl_benchmark.py       # PDOCRL benchmark wrapper
├── compare_results.py              # Results comparison tool
├── check_benchmark_status.py       # Status monitoring
└── train_all_tasks.py              # Legacy benchmark script
```

## Individual Training/Evaluation

### Train a single task
```bash
# Using default config
python train/train_pdocrl.py --task OfflineCarCircle-v0 --device cpu

# Override parameters
python train/train_pdocrl.py \
    --task OfflineCarCircle-v0 \
    --device cuda:0 \
    --update_steps 50000 \
    --actor_lr 1e-4
```

### Evaluate a trained model
```bash
python eval/eval_pdocrl.py \
    --task OfflineCarCircle-v0 \
    --path logs/OfflineCarCircle-v0-cost-10/PDOCRL_<params>/checkpoint_best.pt
```

## Documentation

- **Getting started:** [GETTING_STARTED.md](../GETTING_STARTED.md)
- **Quick reference:** [QUICKSTART_BENCHMARK.md](../QUICKSTART_BENCHMARK.md)
- **Full guide:** [BENCHMARK_README.md](../BENCHMARK_README.md)
- **Implementation details:** [IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md)

## Available Algorithms

| Algorithm | Train | Eval | Config |
|-----------|-------|------|--------|
| BC | `train/train_bc.py` | `eval/eval_bc.py` | `configs/bc_configs.py` |
| BCQL | `train/train_bcql.py` | `eval/eval_bcql.py` | `configs/bcql_configs.py` |
| BEARL | `train/train_bearl.py` | `eval/eval_bearl.py` | `configs/bearl_configs.py` |
| CDT | `train/train_cdt.py` | `eval/eval_cdt.py` | `configs/cdt_configs.py` |
| COptiDICE | `train/train_coptidice.py` | `eval/eval_coptidice.py` | `configs/coptidice_configs.py` |
| CPQ | `train/train_cpq.py` | `eval/eval_cpq.py` | `configs/cpq_configs.py` |
| PDOCRL | `train/train_pdocrl.py` | `eval/eval_pdocrl.py` | `configs/pdocrl_configs.py` |

## All 38 Tasks

### BulletSafetyGym (8 tasks)
- OfflineAntCircle-v0, OfflineAntRun-v0, OfflineCarCircle-v0, OfflineDroneCircle-v0
- OfflineDroneRun-v0, OfflineBallCircle-v0, OfflineBallRun-v0, OfflineCarRun-v0

### SafetyGymnasium Car (8 tasks)
- OfflineCarButton1Gymnasium-v0, OfflineCarButton2Gymnasium-v0
- OfflineCarCircle1Gymnasium-v0, OfflineCarCircle2Gymnasium-v0
- OfflineCarGoal1Gymnasium-v0, OfflineCarGoal2Gymnasium-v0
- OfflineCarPush1Gymnasium-v0, OfflineCarPush2Gymnasium-v0

### SafetyGymnasium Point (8 tasks)
- OfflinePointButton1Gymnasium-v0, OfflinePointButton2Gymnasium-v0
- OfflinePointCircle1Gymnasium-v0, OfflinePointCircle2Gymnasium-v0
- OfflinePointGoal1Gymnasium-v0, OfflinePointGoal2Gymnasium-v0
- OfflinePointPush1Gymnasium-v0, OfflinePointPush2Gymnasium-v0

### SafetyGymnasium Velocity (5 tasks)
- OfflineAntVelocityGymnasium-v1, OfflineHalfCheetahVelocityGymnasium-v1
- OfflineHopperVelocityGymnasium-v1, OfflineSwimmerVelocityGymnasium-v1
- OfflineWalker2dVelocityGymnasium-v1

### MetaDrive (9 tasks)
- OfflineMetadrive-easysparse-v0, OfflineMetadrive-easymean-v0, OfflineMetadrive-easydense-v0
- OfflineMetadrive-mediumsparse-v0, OfflineMetadrive-mediummean-v0, OfflineMetadrive-mediumdense-v0
- OfflineMetadrive-hardsparse-v0, OfflineMetadrive-hardmean-v0, OfflineMetadrive-harddense-v0

## Help

For any script:
```bash
python <script_name>.py --help
```
