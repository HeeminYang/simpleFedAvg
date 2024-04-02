# simple FedAvg

연합학습의 기본이되는 [Federated averaging (FedAvg)](https://arxiv.org/abs/1602.05629) 논문을 코드 구현하였다.
Code implementation of the paper [Federated averaging (FedAvg)](https://arxiv.org/abs/1602.05629), which is the basis of federated learning.

## Pre-requisites
Run the following commands to clone this repository and install the required packages.
```bash
git clone https://github.com/HeeminYang/simpleFedAvg.git
```

## Hyperparameters
- **num_node**: The total number of client in the federated learning.
- **total_round**: The number of round on federated learning.

## Run Experiment
- Run the following command to run the experiment.
```bash
python main.py --config=./exps/hider.json
```
