import numpy as np
import itertools
import pandas as pd
from Layer import Input, Layer
from NeuralNetwork import NeuralNetwork
from time import time
import pickle

def get_data(filename):
    """Get data from filename.

    Args:
        filename (string): name of the file.

    Returns:
        pd.DataFrame: dataframe cointaining data.
    """    
    df = pd.read_csv(filename, names = ['all'])
    df['all'] = df['all'].str.strip()
    df['all1']  = df['all'].apply(lambda x: np.array(x.split(' ')))
    df[['target','attr1','attr2','attr3','attr4','attr5','attr6','id']] = pd.DataFrame(df.all1.tolist(), index= df.index)
    return df.drop(columns=['all','all1'])

def onehot_encoding(x):
    """ Perform one hot encoding.

    Args:
        x (np.array): input array to modify.

    Returns:
        np.array: updated array.
    """    
    X = []
    for row in x:
        unique = np.unique(row)
        for i in unique:
            X.append((row == i).astype(int))

    return np.array(X)

def makeGrid(param_dict):
    """ Produce list with every combination of param grid.

    Args:
        param_dict (dict): dictionary with parameters.

    Returns:
        list: list with every combination of parameters.
    """    
    keys = param_dict.keys()
    combinations = itertools.product(*param_dict.values())
    ds = [dict(zip(keys,cc)) for cc in combinations]
    return ds

def cross_validation(input, target, folds, metrics, params, callbacks):
    """ Perform a k-fold cross-validation.

    Args:
        input (np.array): input array.
        target (np.array): target array.
        folds (int): number of folds.
        metrics (list): list with metrics.
        params (dict): dictionary with parameters to try.
        callbacks (dict): dictionary containing information about callbacks.

    Returns:
        dict: dictionary containing training anda validation information.
    """    

    dim_input = input.shape[1]
    input_index = np.arange(dim_input)
    np.random.shuffle(input_index)
    subset_index = []
    
    loss = params.pop('loss')
    scale_eta_batchsize = params.pop('scale_eta_batchsize')
    output_act_func = params.pop('output_act_func')
    dim_hidden = [params.pop('dim_hidden')]
    hidden_act_func = [params.pop('hidden_act_func')]

    o = 2
    while True:
        if f'dim_hidden{o}' in params.keys():
            dim_hidden.append(params.pop(f'dim_hidden{o}'))
            hidden_act_func.append(params.pop(f'hidden_act_func{o}'))
            o += 1
        else: break
            
    init_eta = params['eta']

    if params['n_batch'] == 'batch':
        train_batch = 1
    else:
        train_batch = 0
        
    for i in range(folds-1):
        subset_index.append(input_index[i*np.round(dim_input / folds).astype(int): (i+1)*np.round(dim_input / folds).astype(int)])
    subset_index.append(input_index[(i+1)*np.round(dim_input / folds).astype(int):])

   
    history_cv = {'train_loss': [],
                  #'train_loss_var': [],
                  'val_loss': []
                  #'val_loss_var': []
                  }
    
    for m in metrics:
        history_cv[f'train_{m.__name__}'] = []
        #history_cv[f'train_{m.__name__}_var'] = []
        history_cv[f'val_{m.__name__}'] = []
        #history_cv[f'val_{m.__name__}_var'] = []

    for val_ind in subset_index:

        train_ind = list(set(input_index) - set(val_ind))

        if train_batch:
            params['n_batch'] = len(train_ind)

        if scale_eta_batchsize == 'lin':
            params['eta'] = init_eta * params['n_batch']
        if scale_eta_batchsize == 'sqrt':
            params['eta'] = init_eta * np.sqrt(params['n_batch'])

        train_input = input[:,train_ind]
        train_target = target[:,train_ind]
        val_input = input[:,val_ind]
        val_target = target[:,val_ind]

        input_layer = Input(train_input.shape[0])
        hidden_layer = Layer(input_layer, dim_hidden[0], hidden_act_func[0])
        for o in range(len(dim_hidden)-1):
            hidden_layer = Layer(hidden_layer, dim_hidden[o+1], hidden_act_func[o+1])
        #hidden_layer = Layer(hidden_layer, dim_hidden2, hidden_act_func2)
        output_layer = Layer(hidden_layer, train_target.shape[0], output_act_func)

        model = NeuralNetwork(input_layer, output_layer, loss, metrics = metrics)

        history = model.train(train_input, train_target, **params,
                              **callbacks,
                                validation_data = [val_input, val_target],
                                verbose=0
                            )
        
        if history == -1:
            history_cv['train_loss_mean'] = np.nan
            history_cv['train_loss_std'] = np.nan
            history_cv['val_loss_mean'] = np.nan
            history_cv['val_loss_std'] = np.nan
            del history_cv['train_loss']
            del history_cv['val_loss']

            for m in metrics:
                history_cv[f'train_{m.__name__}_mean'] = np.nan
                history_cv[f'train_{m.__name__}_std'] = np.nan
                history_cv[f'val_{m.__name__}_mean'] = np.nan
                history_cv[f'val_{m.__name__}_std'] = np.nan
                del history_cv[f'train_{m.__name__}']
                del history_cv[f'val_{m.__name__}']

            return history_cv
        
        history_cv['train_loss'].append(history['train_loss'][-1])
        history_cv['val_loss'].append(history['val_loss'][-1])
        for m in metrics:
            history_cv[f'train_{m.__name__}'].append(history[f'train_{m.__name__}'][-1])
            history_cv[f'val_{m.__name__}'].append(history[f'val_{m.__name__}'][-1])
    
    history_cv['train_loss_mean'] = np.mean(history_cv['train_loss'])
    history_cv['train_loss_std'] = np.std(history_cv['train_loss'])
    history_cv['val_loss_mean'] = np.mean(history_cv['val_loss'])
    history_cv['val_loss_std'] = np.std(history_cv['val_loss'])
    del history_cv['train_loss']
    del history_cv['val_loss']

    for m in metrics:
        history_cv[f'train_{m.__name__}_mean'] = np.mean(history_cv[f'train_{m.__name__}'])
        history_cv[f'train_{m.__name__}_std'] = np.std(history_cv[f'train_{m.__name__}'])
        history_cv[f'val_{m.__name__}_mean'] = np.mean(history_cv[f'val_{m.__name__}'])
        history_cv[f'val_{m.__name__}_std'] = np.std(history_cv[f'val_{m.__name__}'])
        del history_cv[f'train_{m.__name__}']
        del history_cv[f'val_{m.__name__}']

    return history_cv

def grid_search(input, target, params, cv_folds, metrics, callbacks):
    """ Perform a grid search given parameter combination list.

    Args:
        input (np.array): input array.
        target (np.array): target array.
        params (dict): dictionary with list of trials for every parameter.
        cv_folds (int): number of folds for the cross validation.
        metrics (list): list of metrics.
        callbacks (dict): dictionary containing information about callbacks.

    Returns:
        dict,list: dictionary with best set of parameters and its results, list with every combination and its results.
    """    
    param_grid = makeGrid(params)
    print('Starting grid_search...')
    print(f'Grid of parameters: {params}')
    print('-------------------------------------------------')
    for i,p_comb in enumerate(param_grid):
        
        print(f'Starting params {i+1}/{len(param_grid)}: {p_comb}')
        t0 = time()
        p_comb_copy  = p_comb.copy()
        p_comb['results'] = cross_validation(input, target, cv_folds, metrics, p_comb_copy, callbacks)
        p_comb['elapsed_time'] = time() - t0
        print(f'Results:')
        for key, value in p_comb['results'].items():
            print(f'{key}: {value:.2e}')
        print(f"Elapsed time: {p_comb['elapsed_time']:2f} s")
        print('-------------------------------------------------')
    
    best_m = param_grid[0]['results'][f'val_loss_mean']
    best_comb = param_grid[0]
    for p_comb in param_grid:
        if p_comb['results'][f'val_loss_mean'] < best_m:
            best_m = p_comb['results'][f'val_loss_mean']
            best_comb = p_comb
    print(f'Best combination of parameters: {best_comb}')
    return best_comb, param_grid

def save_dict_to_file(dictionary, filename):
    """ Save dictionary to file using pickle

    Args:
        dictionary (dict): dictionary to save.
        filename (str): name of the file.
    """
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

def load_dict_from_file(filename):
    """Load a dictionary from a file using pickle.

    Args:
        filename (str): name of the file.

    Returns:
        dict: loaded dict.
    """
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict
