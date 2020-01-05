# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:07:16 2019

@author: hcji
"""

import numpy as np
import pandas as pd

from DeepCCS.utils import *
from DeepCCS.model.DeepCCS import DeepCCSModel
from DeepCCS.model.encoders import AdductToOneHotEncoder, SmilesToOneHotEncoder
from DeepCCS.model.splitter import SMILESsplitter
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from matplotlib import pyplot

'''
# if only use CPU
import tensorflow as tf

num_cores = 4
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )
session = tf.Session(config=config)
K.set_session(session)
'''


def split_dataset(dataset, ratio):
    # one SMILES should not be in both train and test dataset
    smiles = dataset['SMILES']
    smiles_unique = np.unique(smiles)
    np.random.shuffle(smiles_unique)
    n = int(ratio * len(smiles_unique))
    train_smiles, test_smiles = smiles_unique[n:], smiles_unique[:n]
    train_index = np.where([i in train_smiles for i in smiles])[0]
    test_index = np.where([i in test_smiles for i in smiles])[0]
    dataset_1 = dataset.loc[train_index,:]
    dataset_2 = dataset.loc[test_index,:]
    dataset_1 = dataset_1.reset_index(drop=True)
    dataset_2 = dataset_2.reset_index(drop=True)
    return dataset_1, dataset_2


if __name__ == "__main__":
    
    dataset = pd.read_csv('Data/data.csv')
    
    # remove compounds if the smiles have a '.'
    keep = np.where(['.' not in i for i in dataset['SMILES']])[0]
    dataset = dataset.loc[keep,:]
    dataset = dataset.reset_index(drop=True)
    
    # split dataset
    train_set, test_set = split_dataset(dataset, 0.1)
    train_set, valid_set = split_dataset(train_set, 0.11)
    
    # encoder
    smiles_encoder = SmilesToOneHotEncoder()
    smiles_encoder.fit(dataset['SMILES'])
    train_smi = smiles_encoder.transform(train_set['SMILES'])
    valid_smi = smiles_encoder.transform(valid_set['SMILES'])

    adducts_encoder = AdductToOneHotEncoder()
    adducts_encoder.fit(dataset['Adducts'])
    train_add = adducts_encoder.transform(train_set['Adducts'])
    valid_add = adducts_encoder.transform(valid_set['Adducts'])
    
    # train model
    model = DeepCCSModel()
    model.adduct_encoder = adducts_encoder
    model.smiles_encoder = smiles_encoder
    model.create_model()
    model.model.summary()
    
    m_checkpoint = ModelCheckpoint('Output/DeepCCS/model.h5', save_best_only=True, save_weights_only=True)
    
    model.train_model(X1_train=train_smi, X2_train=train_add, Y_train=train_set['CCS'],
                      X1_valid=valid_smi, X2_valid=valid_add, Y_valid=valid_set['CCS'],
                      model_checkpoint=m_checkpoint, nbr_epochs=100, verbose=1)
    
    # test model
    model = DeepCCSModel()
    model.adduct_encoder = adducts_encoder
    model.smiles_encoder = smiles_encoder
    model.create_model()
    model.model.load_weights('Output/DeepCCS/model.h5')
    model._is_fit = True
    predictions = model.predict(test_set['SMILES'], test_set['Adducts']).flatten()
    
    r2 = r2_score(y_true=test_set['CCS'], y_pred=predictions)
    mae = mean_absolute_error(y_true=test_set['CCS'], y_pred=predictions)
    rmae = np.mean(np.abs(predictions - test_set['CCS']) / test_set['CCS']) * 100
    