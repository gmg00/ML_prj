import numpy as np
import itertools
import pandas as pd
from Layer import Layer, Input
from NeuralNetwork import NeuralNetwork

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
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """    
    X = []
    for row in x:
        unique = np.unique(row)
        for i in unique:
            X.append((row == i).astype(int))

    return np.array(X)

def makeGrid(param_dict):  
    keys = param_dict.keys()
    combinations = itertools.product(*param_dict.values())
    ds = [dict(zip(keys,cc)) for cc in combinations]
    return ds

def cross_validation(input, target, folds, metrics, params, callbacks):

    dim_input = input.shape[1]
    input_index = np.arange(dim_input)
    np.random.shuffle(input_index)
    subset_index = []
    
    scale_eta_batchsize = params.pop('scale_eta_batchsize')
    #dim_hidden = params.pop('dim_hidden')
    #hidden_act_func = params.pop('hidden_act_func')
    struct = params.pop('struct')
    #dim_hidden, hidden_act_func = zip(struct)

    if scale_eta_batchsize == 'lin':
        params['eta'] = params['eta'] * params['n_batch']
    if scale_eta_batchsize == 'sqrt':
        params['eta'] = params['eta'] * np.sqrt(params['n_batch'])
        
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

        train_input = input[:,train_ind]
        train_target = target[:,train_ind]
        val_input = input[:,val_ind]
        val_target = target[:,val_ind]


        ###MODIFICA: AGGIUNTO CICLO FOR
        # For per aumentare hidden layers
        input_layer = Input(17)
        for i, s in enumerate(struct):
            #zip(dim_hidden, hidden_act_func)):
            dim, act_func = s
            if i == 0:
                hidden_layer = Layer(input_layer, dim, act_func)
            else:
                hidden_layer = Layer(hidden_layer, dim, act_func)
        
                
        output_layer = Layer(hidden_layer, 1, 'sigm')

        model = NeuralNetwork(input_layer, output_layer, loss = 'binary_crossentropy', metrics = metrics)

        history = model.train(train_input, train_target, **params,
                              **callbacks,
                                validation_data = [val_input, val_target],
                                verbose=1
                            )
        
        history_cv['train_loss'].append(history['train_loss'][-1])
        history_cv['val_loss'].append(history['val_loss'][-1])
        for m in metrics:
            history_cv[f'train_{m.__name__}'].append(history[f'train_{m.__name__}'][-1])
            history_cv[f'val_{m.__name__}'].append(history[f'val_{m.__name__}'][-1])
    
    history_cv['train_loss_mean'] = np.mean(history_cv['train_loss'])
    history_cv['train_loss_std'] = np.std(history_cv['train_loss'])
    history_cv['val_loss_mean'] = np.mean(history_cv['val_loss'])
    history_cv['val_loss_std'] = np.std(history_cv['val_loss'])
    #del history_cv['train_loss']
    #del history_cv['val_loss']

    for m in metrics:
        history_cv[f'train_{m.__name__}_mean'] = np.mean(history_cv[f'train_{m.__name__}'])
        history_cv[f'train_{m.__name__}_std'] = np.std(history_cv[f'train_{m.__name__}'])
        history_cv[f'val_{m.__name__}_mean'] = np.mean(history_cv[f'val_{m.__name__}'])
        history_cv[f'val_{m.__name__}_std'] = np.std(history_cv[f'val_{m.__name__}'])
        #del history_cv[f'train_{m.__name__}']
        #del history_cv[f'val_{m.__name__}']

    return history_cv

def grid_search(input, target, params, cv_folds, metrics, callbacks):
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
    
    best_m = param_grid[0]['results'][f'val_accuracy_mean']
    best_comb = param_grid[0]
    for p_comb in param_grid:
        if p_comb['results'][f'val_accuracy_mean'] > best_m:
            best_m = p_comb['results'][f'val_accuracy_mean']
            best_comb = p_comb
    print(f'Best combination of parameters: {best_comb}')
    return best_comb, param_grid

def random_search(input, target, params, cv_folds, metrics, callbacks, tries):
    param_grid = makeGrid(params)
    index = np.arange(len(param_grid))
    np.random.shuffle(index)
    print('Starting grid_search...')
    print(f'Grid of parameters: {params}')
    print('-------------------------------------------------')
    param_grid = param_grid[index]
    for p_comb in param_grid[:tries]:
        print(f'Starting params: {p_comb}')
        p_comb['results'] = cross_validation(input, target, cv_folds, metrics, p_comb, callbacks)
        
        print(f'Results:')
        for key, value in p_comb['results'].items():
            print(f'{key}: {value:.2e}')
        print('-------------------------------------------------')
    
    best_m = param_grid[0]['results'][f'val_accuracy_mean']
    best_comb = param_grid[0]
    for p_comb in param_grid:
        if p_comb['results'][f'val_accuracy_mean'] > best_m:
            best_m = p_comb['results'][f'val_accuracy_mean']
            best_comb = p_comb
    print(f'Best combination of parameters: {best_comb}')
    return best_comb

def save_dict_to_file(dictionary, filename):
    """
    Save a dictionary to a file using pickle.

    Parameters:
    - dictionary: The dictionary to be saved.
    - filename: The name of the file to save the dictionary to.
    """
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

def load_dict_from_file(filename):
    """
    Load a dictionary from a file using pickle.

    Parameters:
    - filename: The name of the file containing the dictionary.

    Returns:
    - The loaded dictionary.
    """
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict
