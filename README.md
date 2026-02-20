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
| PDOCRL           | Dual Based Algorithm           | [Offline Constrained Reinforcement Learning under Partial Data Coverage](https://arxiv.org/pdf/2505.17506) |
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
git clone https://github.com/liuzuxin/OSRL.git
cd osrl
pip install -e .
```

If you want to use the `CDT` algorithm, please also manually install the `OApackage`:
```bash
pip install OApackage==2.7.6
```

## How to use OSRL

The example usage are in the `examples` folder, where you can find the training and evaluation scripts for all the algorithms. 
All the parameters and their default configs for each algorithm are available in the `examples/configs` folder. 
OSRL uses the `WandbLogger` in [FSRL](https://github.com/liuzuxin/FSRL) and [Pyrallis](https://github.com/eladrich/pyrallis) configuration system. The offline dataset and offline environments are provided in [DSRL](https://github.com/liuzuxin/DSRL), so make sure you install both of them first.

### Training
For example, to train the `bcql` method, simply run by overriding the default parameters:

```shell
python examples/train/train_bcql.py --task OfflineCarCircle-v0 --param1 args1 ...
```
By default, the config file and the logs during training will be written to `logs\` folder and the training plots can be viewed online using Wandb.

You can also launch a sequence of experiments or in parallel via the [EasyRunner](https://github.com/liuzuxin/easy-runner) package, see `examples/train_all_tasks.py` for details.

### Evaluation
To evaluate a trained agent, for example, a BCQ agent, simply run
```shell
python examples/eval/eval_bcql.py --path path_to_model --eval_episodes 20
```
It will load config file from `path_to_model/config.yaml` and model file from `path_to_model/checkpoints/model.pt`, run 20 episodes, and print the average normalized reward and cost. The pretrained checkpoints for all datasets are available [here](https://drive.google.com/drive/folders/1lZmw2NVNR4YGUdrkih9o3rTMDrWCI_jw?usp=sharing) for reference.
