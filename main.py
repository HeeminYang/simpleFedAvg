import sys
import os
import copy
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, TensorDataset
from efficientnet_pytorch import EfficientNet
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import OrderedDict
from process import *
from agent import agent
from data import get_loader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 128),  # 입력 데이터의 크기를 10으로 가정
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)   # 출력 클래스의 수를 2로 가정
        )
        
    def forward(self, x):
        return self.layers(x)

def create_agent(local_loader_list, common_loader_list, num_node):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP()
    agent_list = []
    for i in range(1,num_node + 1):
        setattr(mod, f'agent{i}', eval(f'agent(copy.deepcopy(model), local_loader_list[{i-1}], common_loader_list[{i-1}], "agent{i}", device)'))
        eval(f'agent_list.append(agent{i})')
    return agent_list

def get_logger(path):

    train_logger = logging.getLogger('train file')
    # label_logger = logging.getLogger('label file')

    stream_handler = logging.StreamHandler()
    train_handler = logging.FileHandler(path)
    # label_handler = logging.FileHandler(f'./log/label_{args.num_class}_{args.num_baby}_{args.sitting_round}.txt')

    train_logger.setLevel(logging.INFO)
    # label_logger.setLevel(logging.INFO)

    train_logger.addHandler(stream_handler)
    train_logger.addHandler(train_handler)
    # label_logger.addHandler(stream_handler)
    # label_logger.addHandler(label_handler)
    return train_logger #, label_logger

# python3 main.py --num_node 5 --total_round 50 --epoch_per_round 1 --num_class 20 --sample_rate 1.0 --batch_size 64 --seed 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_node', type = int)
    parser.add_argument('--total_round', type=int)
    parser.add_argument('--epoch_per_round', type = int)
    parser.add_argument('--num_class', type=int)
    parser.add_argument('--sample_rate', type=float)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    mod = sys.modules[__name__]

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists('./result/log/'):
        os.makedirs('./result/log/')
    if not os.path.exists('./result/img/'):
        os.makedirs('./result/img/')

    now = time.strftime('%Y%m%d%H%M%S')
    log_path = f'./result/log/{now}_epoch{args.epoch_per_round}_sample{args.sample_rate}.txt'
    
    if os.path.exists(log_path):
        os.remove(log_path)
    train_logger = get_logger(log_path)
         
    local_loader_list = [0,0,0,0,0]
    # for data in local_data_dir:
    #     # CSV 파일 로드
    #     dataframe = pd.read_csv(data)

    #     # 데이터와 레이블 분리 (이 부분은 실제 데이터에 따라 변경해야 합니다)
    #     # 예를 들어, 'features'는 특성 열의 이름들의 리스트이고, 'label'은 레이블 열의 이름입니다.
    #     features = dataframe.iloc[:,:-1].values
    #     labels = dataframe.iloc[:,-1].values

    #     # numpy 배열을 PyTorch 텐서로 변환
    #     features_tensor = torch.tensor(features, dtype=torch.float)
    #     labels_tensor = torch.tensor(labels, dtype=torch.long)
        
    #     # TensorDataset 생성
    #     dataset = TensorDataset(features_tensor, labels_tensor)

    #     # DataLoader 생성
    #     loader = DataLoader(dataset, batch_size=32, shuffle=True)

    #     local_loader_list.append(loader)

    common_loader_list, test_loader = get_loader()

    agent_list = create_agent(local_loader_list, common_loader_list, args.num_node)

    fed_avg(agent_list, args.total_round, args.epoch_per_round, args.sample_rate, args.num_class, test_loader, train_logger)
