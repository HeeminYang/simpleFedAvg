import os
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, random_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

class agent():
    def __init__(self, model, local_dataloader, common_dataloader, client_name, device):
        
        self.name = client_name         # Save node's name for logging.
        self.poor = False               # Set False at initialization.
        self.sitted_round = 0
        self.local_dataloader = local_dataloader
        self.common_dataloader = common_dataloader
        self.sample_size = len(self.common_dataloader.dataset)
        self.device = device

        self.model = nn.DataParallel(model).to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, weight_decay=0.0004)
        self.criterion = nn.CrossEntropyLoss()

    def common_update(self):
        
        # Initiazie poor state every round
        

        self.model.train()
        

        for data, label in self.common_dataloader:

            data = data.to(self.device)
            label = label.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(data)
            label = label
            
            loss = self.criterion(output, label)
            

            loss.backward()
            
            self.optimizer.step()
    
    def local_update(self):
        
        # Initiazie poor state every round
        
        self.model.train()
        

        for data, label in self.local_dataloader:

            data = data.to(self.device)
            label = label.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(data)
            label = label
            
            loss = self.criterion(output, label)
            

            loss.backward()
            
            self.optimizer.step()

    def local_test(self, test_loader):
        
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
              
        for batch_idx, (data, labels) in enumerate(test_loader):

            data = data.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(data)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()
            
            _, pred_labels = torch.max(outputs, 1)
            
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        
        self.accuracy = correct/total

        return self.accuracy, loss/total
    
    def final_test(self, test_loader):
        
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        outputs_list = []
        labels_list = []

        for batch_idx, (data, labels) in enumerate(test_loader):

            data = data.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(data)
            outputs_list.append(outputs)
            labels_list.append(labels)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()
            
            _, pred_labels = torch.max(outputs, 1)
            
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        
        self.accuracy = correct/total

        return self.accuracy, loss/total, outputs_list, labels_list

