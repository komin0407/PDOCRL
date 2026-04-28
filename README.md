## Structure
The structure of this repo is as follows:
```
├── examples
│   ├── configs  # the training configs of each algorithm
│   ├── eval     # the evaluation escipts
│   ├── train    # the training scipts
├── osrl
│   ├── algorithms  # offline safe RL algorithms
│   ├── common      # base networks and utils
```
The implemented offline safe RL and imitation learning algorithms include:

| Algorithm           | Type           | Description           |
|:-------------------:|:-----------------:|:------------------------:|
| BCQ-Lag             | Q-learning           | [BCQ](https://arxiv.org/pdf/1812.02900.pdf) with [PID Lagrangian](https://arxiv.org/abs/2007.03964) |
| BEAR-Lag            | Q-learning           | [BEARL](https://arxiv.org/abs/1906.00949) with [PID Lagrangian](https://arxiv.org/abs/2007.03964)   |
| CPQ                 | Q-learning           | [Constraints Penalized Q-learning (CPQ))](https://arxiv.org/abs/2107.09003) |
| COptiDICE           | Distribution Correction Estimation           | [Offline Constrained Policy Optimization via stationary DIstribution Correction Estimation](https://arxiv.org/abs/2204.08957) |
| PDOCRL           | Dual Based Algorithm           |  |
| CDT                 | Sequential Modeling | [Constrained Decision Transformer](https://arxiv.org/abs/2302.07351) |
| BC-All                 | Imitation Learning | [Behavior Cloning](https://arxiv.org/abs/2302.07351) with all datasets |
| BC-Safe                 | Imitation Learning | [Behavior Cloning](https://arxiv.org/abs/2302.07351) with safe trajectories |
| BC-Frontier                 | Imitation Learning | [Behavior Cloning](https://arxiv.org/abs/2302.07351) with high-reward trajectories |


## Installation

OSRL is currently hosted on [PyPI](https://pypi.org/project/osrl-lib), you can simply install it by:

```bash
pip install osrl-lib
```

You can also pull the repo and install:
```bash
git clone https://github.com/komin0407/PDOCRL.git
cd PDOCRL
pip install -e .

```
The offline datasets and environments are provided by [DSRL](https://github.com/liuzuxin/DSRL). Install it as well:
```bash
pip install dsrl
```

If you want to use the `CDT` algorithm, please also manually install the `OApackage`:
```bash
pip install OApackage==2.7.6
```
## How to use

The example scripts are in the `examples` folder. All parameters and their default configs are in `examples/configs/`. This repo uses [Pyrallis](https://github.com/eladrich/pyrallis) for configuration and [WandbLogger](https://github.com/liuzuxin/FSRL) for logging.

### Training PDOCRL
```shell
python examples/train/train_pdocrl.py --task OfflineCarCircle-v0 --cost_limit 10 --device cpu
```

### Training baselines
For example, to train `bcql`:
```shell
python examples/train/train_bcql.py --task OfflineCarCircle-v0 --param1 args1 ...
```
Config files and logs during training are written to the `logs/` folder. Training plots can be viewed online via Wandb.

You can also run all tasks in parallel:
```shell
python examples/train_all_tasks.py
```

### Evaluation
To evaluate a trained PDOCRL agent:
```shell
python examples/eval/eval_pdocrl.py --path path_to_model --eval_episodes 20
```

To evaluate a baseline, e.g. BCQ-Lag:
```shell
python examples/eval/eval_bcql.py --path path_to_model --eval_episodes 20
```

Each eval script loads `path_to_model/config.yaml` and `path_to_model/checkpoints/model.pt`, runs the specified number of episodes, and prints the average normalized reward and cost.
