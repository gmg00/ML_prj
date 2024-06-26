import numpy as np
from matplotlib import pyplot as plt
from NeuralNetwork import NeuralNetwork
from utils import get_data, onehot_encoding, grid_search, save_dict_to_file, load_dict_from_file
from Layer import Layer, Input
from functions import accuracy, MSE, MEE
import pandas as pd

if __name__ == '__main__':
    
    do_grid_search = True
<<<<<<< HEAD
    retraining_epochs = 500

    best_comb_filename = '/mnt/c/Users/HP/Desktop/UNI/LM_1/MachineLearning/ML_prj/data/output/best_comb_cup12.pkl'
    param_grid_filename = '/mnt/c/Users/HP/Desktop/UNI/LM_1/MachineLearning/ML_prj/data/output/param_grid_cup12.pkl'
=======
    retraining_epochs = 1000

    best_comb_filename = '/mnt/c/Users/HP/Desktop/UNI/LM_1/MachineLearning/ML_prj/data/output/best_comb_cup3.pkl'
    param_grid_filename = '/mnt/c/Users/HP/Desktop/UNI/LM_1/MachineLearning/ML_prj/data/output/param_grid_cup3.pkl'
>>>>>>> main
    # DATASET ACQUISITION
    
    names = ['id', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 
         'feature_7', 'feature_8', 'feature_9', 'feature_10', 'target_x', 'target_y','target_z']

    df = pd.read_csv('/mnt/c/Users/HP/Desktop/UNI/LM_1/MachineLearning/ML_prj/data/ML-CUP23-TR.csv', names=names, comment='#')

    targets = ['target_x', 'target_y', 'target_z']
    features = list(set(names) - {'id', 'target_x', 'target_y', 'target_z'})

    df = df.sample(frac=1)
    prova = df[:750]
    X_train, y_train = prova[features].to_numpy().T, prova[targets].to_numpy().T
    X_test, y_test = df[750:][features].to_numpy().T, df[750:][targets].to_numpy().T

    # GRID SEARCH

    # HYPERPARAMETERS DICTONARY
    if do_grid_search:
        params = {
<<<<<<< HEAD
              'eta' : [0.008,0.01],
              'lam' : [0.00003,0.00001],
              'alpha':[0.9],
=======
              'eta' : [0.009,0.007,0.005],
              'lam' : [0.00001,0.01],
              'alpha':[0.9,0.5],
>>>>>>> main
              'epochs': [500],
              'n_batch' : [150],
              'scale_eta_batchsize' : [None], #'sqrt' per eta * sqrt(n_batch), 'lin' per eta * n_batch
              
<<<<<<< HEAD
              'dim_hidden' : [70],
              'hidden_act_func' : ['tanh'],
              'dim_hidden2' : [70],
              'hidden_act_func2' : ['tanh'],
=======
              'dim_hidden' : [50],
              'hidden_act_func' : ['tanh'],
              'dim_hidden2' : [50],
              'hidden_act_func2' : ['tanh'],
              'dim_hidden3' : [50],
              'hidden_act_func3':['tanh'],
>>>>>>> main

              'use_opt' : [1],
              'loss' : ['MSE'],
              'output_act_func' : ['lin'],
<<<<<<< HEAD
              'nest': [True]
=======
              'nest' : [False,True],
              'l1_reg':[False]

>>>>>>> main
            }
        
        callbacks = {
            'early_stopping' : None,
            'reduce_eta' : None
        }
        
        best_comb, param_grid = grid_search(X_train, y_train.reshape((3,X_train.shape[1])), params, 5, [MEE], callbacks)

        save_dict_to_file(best_comb,best_comb_filename)
        save_dict_to_file(param_grid,param_grid_filename)

    best_comb = load_dict_from_file(best_comb_filename)

    print(best_comb)

    #results = best_comb.pop('results')
    #elapsed_time = best_comb.pop('elapsed_time')

    best_comb.pop('scale_eta_batchsize')

    if best_comb['n_batch'] == 'batch':
        best_comb['n_batch'] = X_train.shape[1]

    best_comb['epochs'] = retraining_epochs
    
    input_layer = Input(X_train.shape[0])
    hidden_layer = Layer(input_layer, best_comb.pop('dim_hidden'), best_comb.pop('hidden_act_func'))
    o = 2
    while True:
        if f'dim_hidden{o}' in best_comb.keys():
            hidden_layer = Layer(hidden_layer, best_comb.pop(f'dim_hidden{o}'), best_comb.pop(f'hidden_act_func{o}'))
            o += 1
        else: break
    output_layer = Layer(hidden_layer, 3, best_comb.pop('output_act_func'))

    model = NeuralNetwork(input_layer, output_layer, best_comb.pop('loss'), metrics=[MEE])
    history = model.retrain(X_train, y_train.reshape((3,X_train.shape[1])), test_data = [X_test,y_test.reshape((3,X_test.shape[1]))], **best_comb)

    plt.figure(1)
    plt.plot(history['train_loss'],label='train_loss')
    plt.plot(history['test_loss'], label='test_loss')
    plt.yscale('log')
    plt.xlabel('Epochs', size=15)
    plt.ylabel('Loss functions', size=15)
    plt.title('train_loss vs test_loss', size=18)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.grid()
    plt.legend(fontsize=15)

    plt.figure(2)
    plt.plot(history['train_MEE'],label='train_MEE')
    plt.plot(history['test_MEE'], label='test_MEE')
    plt.xlabel('Epochs', size=15)
    plt.ylabel('MEE', size=15)
    plt.title('train_MEE vs test_MEE', size=18)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.grid()
    plt.legend(fontsize=15)

    plt.show()
    

    # (Grid Search sui callback) -> MONK_exploration

    # Retrain con migliori iperparametri

    # Error assesment su TS

