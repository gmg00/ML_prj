import numpy as np
from matplotlib import pyplot as plt
from NeuralNetwork import NeuralNetwork
from utils import get_data, onehot_encoding, grid_search
from Layer import Layer, Input
from functions import accuracy
from utils import *

if __name__ == '__main__':

    # DATASET ACQUISITION
    
    df = get_data('/Users/Silvia.S/Desktop/MONKS datasets/monks-3.train.csv')
    df_test = get_data('/Users/Silvia.S/Desktop/MONKS datasets/monks-3.test.csv')

    # Suddivisione TR/VL e TS (80% - 20%)

    X_train, y_train = df.drop(columns=['target','id']).to_numpy().T, df['target'].apply(lambda x: int(x)).to_numpy().T
    X_test, y_test = df_test.drop(columns=['target','id']).to_numpy().T, df_test['target'].apply(lambda x: int(x)).to_numpy().T

    # DATA PREPARATION

    X_train = onehot_encoding(X_train)
    X_test = onehot_encoding(X_test)

    # GRID SEARCH

    # HYPERPARAMETERS DICTONARY

    params = {
          'eta' : [0.01, 0.05, 0.07, 0.1, 0.3],
          'lam' : [0.0],
          'alpha':[0.5, 0.7, 0.9],
          'epochs': [50],
          'n_batch' : [31],
          #'n_batch' : [1,31],
          'scale_eta_batchsize' : ['sqrt'], #'sqrt' per eta * sqrt(n_batch), 'lin' per eta * n_batch
          'struct' : [[(5, 'sigm'), (5, 'sigm')]]   
            }
            '''struct= modo per passare più layers, il primo valore fra tonde è il numero di unità,
          il secondo la act_ func; le due tonde insieme formano due layers'''
    
    early_stopping = {'patience' : 150,
                    'monitor' : 'val_accuracy',
                    'verbose' : 1,
                    'compare_function': np.greater_equal}

    reduce_eta = {'patience' : 75,
                'monitor' : 'val_accuracy',
                'factor' : 0.5,
                'verbose' : 1,
                'compare_function': np.greater_equal}

    callbacks = {'early_stopping': None,
                'reduce_eta': None}

    best_comb = grid_search(X_train, y_train.reshape((1,-1)),
                                   params, 5, [accuracy], callbacks)
    #, best_train_loss, best_val_loss, best_train_acc, best_val_acc

    print(best_comb['results'].keys())
    
    plt.plot(best_comb['results']['train_loss'],label='train_loss')
    plt.plot(best_comb['results']['val_loss'], label='val_loss')
    plt.xlabel('Epochs', size=15)
    plt.ylabel('Loss functions', size=15)
    plt.title('Train_loss vs val_loss', size=18)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.grid()
    plt.legend(fontsize=15)
    plt.show()
    plt.clf()

    plt.plot(best_comb['results']['train_accuracy'],label='train_accuracy')
    plt.plot(best_comb['results']['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epochs', size=15)
    plt.ylabel('Accuracy', size=15)
    plt.title(' Train_accuracy vs val_accuracy', size=18)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.grid()
    plt.legend(fontsize=15)
    plt.show()
    


    # (Grid Search sui callback) -> MONK_exploration

    # Retrain con migliori iperparametri

    # Error assesment su TS
