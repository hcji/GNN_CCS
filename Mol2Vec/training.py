# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:07:16 2019

@author: hcji
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg

'''
# if only use CPU
from keras import backend as K
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
    return dataset_1, dataset_2


if __name__ == "__main__":
    
    from keras.models import Model, load_model
    from keras.layers import Dense, Input
    from keras import metrics, optimizers
    from keras.callbacks.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau 
    from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
    
    dataset = pd.read_csv('Data/data.csv')
    train_set, test_set = split_dataset(dataset, 0.1)
    
    model = word2vec.Word2Vec.load('Mol2Vec/pretrain/model_300dim.pkl')
    train_mol = [Chem.MolFromSmiles(x) for x in train_set['SMILES']]
    test_mol = [Chem.MolFromSmiles(x) for x in test_set['SMILES']]
    
    train_sent = [mol2alt_sentence(x, 1) for x in train_mol]
    test_sent = [mol2alt_sentence(x, 1) for x in test_mol]
    
    train_vec = [DfVec(x).vec for x in sentences2vec(train_sent, model, unseen='UNK')]
    test_vec = [DfVec(x).vec for x in sentences2vec(test_sent, model, unseen='UNK')]
    
    train_vec = np.array(train_vec)
    test_vec = np.array(test_vec)
    
    # train model
    layer_in = Input(shape=(train_vec.shape[1],))
    layer_dense = layer_in
    n_nodes = 32
    for j in range(3):
        layer_dense = Dense(int(n_nodes), activation="relu")(layer_dense)
    layer_output = Dense(1, activation="linear")(layer_dense)
    opt = optimizers.Adam(lr=0.001)
    model = Model(layer_in, layer_output)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    
    # call back
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('Output/Mol2Vec/model.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')
    
    # fit model
    model.fit(train_vec, train_set['CCS'], epochs=50, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.11)
    
    # test
    model = load_model('Output/Mol2Vec/model.h5')
    predictions = model.predict(test_vec)[:,0]
    
    r2 = r2_score(y_true=test_set['CCS'], y_pred=predictions)
    mae = mean_absolute_error(y_true=test_set['CCS'], y_pred=predictions)
    rmae = np.mean(np.abs(predictions - test_set['CCS']) / test_set['CCS']) * 100    
    