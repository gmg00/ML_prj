import numpy as np
from matplotlib import pyplot as plt
from NeuralNetwork import NeuralNetwork
from utils import get_data, onehot_encoding, grid_search
from Layer import Layer, Input
from utils import *

if __name__ == '__main__':

    # DATASET ACQUISITION
    
    df = get_data('/Users/HP/Desktop/UNI/LM_1/MachineLearning/ML_prj/data/MONK/monks-1.train')
    df_test = get_data('/Users/HP/Desktop/UNI/LM_1/MachineLearning/ML_prj/data/MONK/monks-1.test')

    # Suddivisione TR/VL e TS (80% - 20%)

    X_train, y_train = df.drop(columns=['target','id']).to_numpy().T, df['target'].apply(lambda x: int(x)).to_numpy().T
    X_test, y_test = df_test.drop(columns=['target','id']).to_numpy().T, df_test['target'].apply(lambda x: int(x)).to_numpy().T

    # DATA PREPARATION

    X_train = onehot_encoding(X_train)
    X_test = onehot_encoding(X_test)

    # GRID SEARCH

    # HYPERPARAMETERS DICTONARY

    params = {
          'eta' : [0.01, 0.02],
          'lam' : [0.0, 0.1],
          'alpha':[0.5,0.9],
          'epochs': [50],
          'n_batch' : [1,31],
          'scale_eta_batchsize' : ['sqrt'], #'sqrt' per eta * sqrt(n_batch), 'lin' per eta * n_batch
          
          'dim_hidden' : [2,3],
          'hidden_act_func' : ['relu']
        }
    
    best_comb = grid_search(X_train, y_train.reshape((1,124)), params, 5, [accuracy], callbacks)
    


    # (Grid Search sui callback) -> MONK_exploration

    # Retrain con migliori iperparametri

    # Error assesment su TS

