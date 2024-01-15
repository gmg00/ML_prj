import numpy as np
import itertools
import pandas as pd

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