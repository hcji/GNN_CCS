# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:09:53 2019

@author: hcji
"""

import pandas as pd
from DeepCCS.utils import read_dataset

datasets = ['Baker', 'CBM', 'McLean', 'Astarita_pos', 'Astarita_neg', 'MetCCS_train_pos', 'MetCCS_train_neg', 'MetCCS_test_pos', 'MetCCS_test_neg']

for i, d in enumerate(datasets):
    if i == 0:
        data = read_dataset('DATASETS.h5', d)
        data['Source'] = d
    else:
        ndata = read_dataset('DATASETS.h5', d)
        ndata['Source'] = d
        data = pd.concat([data, ndata])

data.to_csv('Data/data.csv')