import time
import numpy as np
import wandb
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import copy
from collections import OrderedDict
import pickle

def global_avg(sampled_list):

    model = OrderedDict()
    total_size = sum([i.sample_size for i in sampled_list])

    for i, cli in enumerate(sampled_list,1):
        update_ratio = cli.sample_size / total_size

        if i == 1:
            for key in cli.model.state_dict().keys():
                model[key] = copy.deepcopy(cli.model.state_dict()[key])*update_ratio
        else:
            for key in cli.model.state_dict().keys():
                model[key] += copy.deepcopy(cli.model.state_dict()[key])*update_ratio 

    return model

def fed_avg(client_list, total_round, epoch_per_round, sample_rate, num_class, valid_loader, train_logger):
    now = time.strftime('%Y%m%d%H%M%S')

    # Make empty lists to save global accuracy & loss
    acc_list = []
    loss_list = []
    best_acc = 0
    num_sampled_clients = max(int(sample_rate * len(client_list)), 1)
    
    train_logger.info(f'-------Horizontal_FL-------')

    for r in range(1, total_round + 1):

        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(len(client_list))], size=num_sampled_clients, replace=False).tolist())


        if r > 1:
            for idx in sampled_client_indices:
                client_list[idx].model.load_state_dict(global_model)
            train_logger.info(f'-------After global update-------')

            global_acc, global_loss = client_list[idx].local_test(valid_loader)

            localtime = time.asctime(time.localtime(time.time()))
            train_logger.info(f'[  {localtime}  ] Global Accuracy : {global_acc*100}, Loss : {global_loss}')
            if global_acc > best_acc:
                best_acc = global_acc
                best_model = copy.deepcopy(global_model)
                best_round = r-1
            acc_list.append(global_acc)
            loss_list.append(global_loss)
            train_logger.info('-'*40)

        # Save starting time.
        # time.asctime(): date to string
        localtime = time.asctime(time.localtime(time.time()))
        
        # Local update
        train_logger.info(f'[  {localtime}  ]Round {r} start.')
        train_logger.info('-'*20)
        train_logger.info('Local Update')
        
        for idx in tqdm(sampled_client_indices, desc='local_update'):
            # if r is not 1:
            for _ in range(epoch_per_round):
                client_list[idx].common_update()

        sampled_list = []
        for idx in sampled_client_indices:
            sampled_list.append(client_list[idx])

        # Global update
        train_logger.info('-'*40)
        train_logger.info('Global update...')
        
        global_model = global_avg(sampled_list)
    for cli in client_list:
        cli.model.load_state_dict(global_model)

    train_logger.info(f'-------After global update-------')

    global_acc, global_loss, outputs_list, labels_list = client_list[idx].final_test(valid_loader)
    
    if global_acc > best_acc:
        best_acc = global_acc
        best_model = copy.deepcopy(global_model)
        best_round = r
        
    torch.save(global_model,f'result/model/{now}_Experiment_FedAvg_latest.pt')
    torch.save(best_model, f'result/model/{now}_Experiment_FedAvg_best.pt')
    
    localtime = time.asctime(time.localtime(time.time()))
    train_logger.info(f'[  {localtime}  ] Global Accuracy : {global_acc*100}, Loss : {global_loss}')
    train_logger.info(f'Best Acc: {best_acc} Best Round: {best_round}')
    acc_list.append(global_acc)
    loss_list.append(global_loss)
    train_logger.info('-'*40)