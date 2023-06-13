import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def get_loader():
    client1 = pd.read_csv('data/local/client1.csv')
    X_1 = client1.drop('fraud_bool',axis=1)
    y_1 = client1['fraud_bool']
    client2 = pd.read_csv('data/local/client2.csv')
    X_2 = client2.drop('fraud_bool',axis=1)
    y_2 = client2['fraud_bool']
    client3 = pd.read_csv('data/local/client3.csv')
    X_3 = client3.drop('fraud_bool',axis=1)
    y_3 = client3['fraud_bool']
    client4 = pd.read_csv('data/local/client4.csv')
    X_4 = client4.drop('fraud_bool',axis=1)
    y_4 = client4['fraud_bool']
    client5 = pd.read_csv('data/local/client5.csv')
    X_5 = client5.drop('fraud_bool',axis=1)
    y_5 = client5['fraud_bool']

    test = pd.read_csv('data/test/client_test_under.csv')
    X_test = test.drop('fraud_bool',axis=1)
    y_test = test['fraud_bool']

    common =  ['prev_address_months_count', 'current_address_months_count',
       'customer_age', 'days_since_request', 'intended_balcon_amount',
       'payment_type', 'bank_branch_count_8w',
       'date_of_birth_distinct_emails_4w', 'employment_status',
       'credit_risk_score', 'housing_status', 'phone_mobile_valid',
       'bank_months_count', 'proposed_credit_limit', 'foreign_request',
       'source', 'session_length_in_minutes', 'device_os',
       'device_distinct_emails_8w', 'month']

    # labelencoding

    object_col = {'payment_type':{'AE':0, 'AD':1, 'AC':2, 'AA':3, 'AB':4},
                    'employment_status':{'CE':0, 'CA':1, 'CB':2, 'CC':3, 'CG':4, 'CD':5, 'CF':6},
                    'housing_status':{'BE':0, 'BF':1, 'BC':2, 'BG':3, 'BA':4, 'BD':5, 'BB':6},
                    'source':{'INTERNET':0, 'TELEAPP':1},
                    'device_os':{'other':0, 'windows':1, 'x11':2, 'linux':3, 'macintosh':4}}

    for col,vals in object_col.items():
        X_1[col] = X_1[col].replace(vals)
        X_2[col] = X_2[col].replace(vals)
        X_3[col] = X_3[col].replace(vals)
        X_4[col] = X_4[col].replace(vals)
        X_5[col] = X_5[col].replace(vals)
        X_test[col] = X_test[col].replace(vals)
    
    X_1 = X_1[common]
    X_2 = X_2[common]
    X_3 = X_3[common]
    X_4 = X_4[common]
    X_5 = X_5[common]
    X_test = X_test[common]

    X_5 = scaler.fit_transform(X_5)
    X_4 = scaler.transform(X_4)
    X_3 = scaler.transform(X_3)
    X_2 = scaler.transform(X_2)
    X_1 = scaler.transform(X_1)
    X_test = scaler.transform(X_test)

    DS5 = TensorDataset(torch.Tensor(X_5), torch.LongTensor(y_5.values))
    DS4 = TensorDataset(torch.Tensor(X_4), torch.LongTensor(y_4.values))
    DS3 = TensorDataset(torch.Tensor(X_3), torch.LongTensor(y_3.values))
    DS2 = TensorDataset(torch.Tensor(X_2), torch.LongTensor(y_2.values))
    DS1 = TensorDataset(torch.Tensor(X_1), torch.LongTensor(y_1.values))
    DS_test = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test.values))

    loader5 = DataLoader(DS5, batch_size=64)
    loader4 = DataLoader(DS4, batch_size=64)
    loader3 = DataLoader(DS3, batch_size=64)
    loader2 = DataLoader(DS2, batch_size=64)
    loader1 = DataLoader(DS1, batch_size=64)
    local_loader_list = [loader1, loader2, loader3, loader4, loader5]
    loader_test = DataLoader(DS_test, batch_size=64)

    return local_loader_list, loader_test