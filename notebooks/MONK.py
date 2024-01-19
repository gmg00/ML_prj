import numpy as np
from matplotlib import pyplot as plt
from NeuralNetwork import NeuralNetwork
from utils import get_data, onehot_encoding, grid_search, save_dict_to_file, load_dict_from_file
from Layer import Layer, Input
from functions import accuracy, MSE

if __name__ == '__main__':
    
    do_grid_search = True
    best_comb_filename = '/mnt/c/Users/HP/Desktop/UNI/LM_1/MachineLearning/ML_prj/data/output/best_comb.pkl'
    param_grid_filename = '/mnt/c/Users/HP/Desktop/UNI/LM_1/MachineLearning/ML_prj/data/output/param_grid.pkl'
    # DATASET ACQUISITION
    
    df = get_data('/mnt/c/Users/HP/Desktop/UNI/LM_1/MachineLearning/ML_prj/data/MONK/monks-1.train')
    df_test = get_data('/mnt/c/Users/HP/Desktop/UNI/LM_1/MachineLearning/ML_prj/data/MONK/monks-1.test')

    # Suddivisione TR/VL e TS (80% - 20%)

    X_train, y_train = df.drop(columns=['target','id']).to_numpy().T, df['target'].apply(lambda x: int(x)).to_numpy().T
    X_test, y_test = df_test.drop(columns=['target','id']).to_numpy().T, df_test['target'].apply(lambda x: int(x)).to_numpy().T

    # DATA PREPARATION

    X_train = onehot_encoding(X_train)
    X_test = onehot_encoding(X_test)

    # GRID SEARCH

    # HYPERPARAMETERS DICTONARY
    if do_grid_search:
        params = {
              'eta' : [0.005, 0.01, 0.05, 0.1, 0.5],
              'lam' : [0.0, 0.1],
              'alpha':[0.5, 0.9, 0.1],
              'epochs': [500],
              'n_batch' : [1,31,'batch'],
              'scale_eta_batchsize' : ['sqrt','lin',None], #'sqrt' per eta * sqrt(n_batch), 'lin' per eta * n_batch
              
              'dim_hidden' : [2,3,4],
              'hidden_act_func' : ['relu']
            }
        
        callbacks = {
            'early_stopping' : None,
            'reduce_eta' : None
        }
        
        best_comb, param_grid = grid_search(X_train, y_train.reshape((1,X_train.shape[1])), params, 5, [accuracy, MSE], callbacks)

        save_dict_to_file(best_comb,best_comb_filename)
        save_dict_to_file(param_grid,param_grid_filename)

    best_comb = load_dict_from_file(best_comb_filename)
    print(best_comb)

    results = best_comb.pop('results')
    if best_comb['n_batch'] == 'batch':
        best_comb['n_batch'] = X_train.shape[1]
    elapsed_time = best_comb.pop('elapsed_time')
    if best_comb['scale_eta_batchsize'] == 'lin':
        best_comb['eta'] = best_comb['eta'] * best_comb['n_batch']
    if best_comb['scale_eta_batchsize'] == 'sqrt':
        best_comb['eta'] = best_comb['eta'] * np.sqrt(best_comb['n_batch'])
    best_comb.pop('scale_eta_batchsize')

    #print(best_comb)
    #print(results)
    
    input_layer = Input(X_train.shape[0])
    hidden_layer = Layer(input_layer, best_comb.pop('dim_hidden'), best_comb.pop('hidden_act_func'))
    output_layer = Layer(hidden_layer, 1, 'sigm')

    model = NeuralNetwork(input_layer, output_layer, 'binary_crossentropy', metrics=[accuracy, MSE])
    history = model.retrain(X_train, y_train.reshape((1,X_train.shape[1])), test_data = [X_test,y_test.reshape((1,X_test.shape[1]))], **best_comb)

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
    plt.plot(history['train_accuracy'],label='train_accuracy')
    plt.plot(history['test_accuracy'], label='test_accuracy')
    plt.xlabel('Epochs', size=15)
    plt.ylabel('Accuracy', size=15)
    plt.title('train_accuracy vs test_accuracy', size=18)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.grid()
    plt.legend(fontsize=15)

    plt.figure(3)
    plt.plot(history['train_MSE'],label='train_MSE')
    plt.plot(history['test_MSE'], label='test_MSE')
    plt.yscale('log')
    plt.xlabel('Epochs', size=15)
    plt.ylabel('Accuracy', size=15)
    plt.title('train_MSE vs test_MSE', size=18)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.grid()
    plt.legend(fontsize=15)

    plt.show()

    # (Grid Search sui callback) -> MONK_exploration

    # Retrain con migliori iperparametri

    # Error assesment su TS

