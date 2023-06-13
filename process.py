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

    # Initialize the global model weights
    # model = copy.deepcopy(sampled_list[0].model)
    # for param in model.parameters():
    #     init.zeros_(param)
    # model = model.state_dict()

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
        
            # elif r is 1:
            #     for _ in range(30):
            #         client_list[idx].local_update()

        sampled_list = []
        for idx in sampled_client_indices:
            # acc, loss = client_list[idx].local_test(valid_loader)
            # train_logger.info(f'Agent {idx + 1} Accuracy : {acc*100}, Loss : {loss}')
            sampled_list.append(client_list[idx])

        # Global update
        train_logger.info('-'*40)
        train_logger.info('Global update...')
        
        global_model = global_avg(sampled_list)
        
    # for idx in tqdm(sampled_client_indices, desc='Virtical_FL'):
    #     # if r is not 1:
    #     for _ in range(epoch_per_round):
    #         client_list[idx].local_update()
    
    # for idx in tqdm(sampled_client_indices, desc='final_test'):
    #     client_list[idx].final_update()

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

    with open('result/outputs_list.pkl', 'wb') as f:
        pickle.dump(outputs_list, f)
    with open('result/labels_list.pkl', 'wb') as f:
        pickle.dump(labels_list, f)  
    with open('result/acc_list.pkl', 'wb') as f:
        pickle.dump(acc_list, f) 
    with open('result/loss_list.pkl', 'wb') as f:
        pickle.dump(loss_list, f) 
    
    # 눈금 조정(적은 라운드, 큰 라운드 모두 맞게)
    if total_round >= 25:
        if total_round >= 100:
            if total_round%10 ==0:
                xticks_mj = int(total_round/10)
                xticks_mi = 5
            else:
                xticks_mj = 25
                xticks_mi = 5
        else:
            xticks_mj = 10
            xticks_mi = 1
    else:
        xticks_mj, xticks_mi = 1, 1

    fig = plt.figure()
    fig.set_facecolor('white')
    ax1 = fig.add_subplot()
    ax1.set_xlim([0, total_round+2])
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('Accuracy')
    ax1.xaxis.set_major_locator(MultipleLocator(xticks_mj))
    ax1.xaxis.set_minor_locator(MultipleLocator(xticks_mi))
    ax1.yaxis.set_major_locator(MultipleLocator(0.05))  
    ax1.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax1.set_xlabel('Round')
    ax1.plot(range(1,total_round+1), acc_list, label = 'Accuracy')
    ax2 = ax1.twinx()
    ax2.set_ylim([0.0, 0.3])
    ax2.set_ylabel('Loss')
    ax2.yaxis.set_major_locator(MultipleLocator(0.05))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax2.plot(range(1,total_round+1), loss_list, label = 'Loss', c='orange')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')

    plt.savefig(f'result/img/{now}_Experiment_FedAvg_class{num_class}_sample{sample_rate}.png', dpi=200, facecolor='#eeeeee')