import numpy as np
import math

#linear activation function 
def linear(x):
    """Linear function

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """    
    return x

def d_linear(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """    
    return 1

#sigmoid activation function
def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def d_sigmoid(x):
    f = sigmoid(x)
    return f * (1 - f)

#ReLu activation function
def relu(x):
    if x > 0: return x
    else: return 0

def d_relu(x):
    if x > 0: return 1
    else: return 0
    

#hyperbolic tangent activation function
def TanH(x):
    return np.tanh(x)

def d_TanH(x):
    return 1 - math.pow(np.tanh(x), 2) 

#Dictionary for the activation functions
act_func = {
    'lin': linear,
    'sigm': sigmoid,
    'relu': relu,
    'tanh': TanH
}

#A second dictionary for their derivatives
d_act_func = {
    'lin': d_linear,
    'sigm': d_sigmoid,
    'relu': d_relu,
    'tanh': d_TanH
}

def d_MSE(layer, target):
    return 2*(layer - target)/target.shape[1]

def MSE(layer, target):
    #print(layer.shape)
    return ((layer - target)**2/target.shape[1]).reshape(layer.shape)

def binary_crossentropy(layer, target):
    return (- (target * np.log(layer) + (1 - target) * np.log(1 - layer))/layer.shape[1]).reshape(layer.shape)

def d_binary_crossentropy(layer, target, epsilon=1e-15):
    layer = np.clip(layer, epsilon, 1 - epsilon)  # Avoid division by zero
    return (layer - target)/(layer*(1-layer)*target.shape[1])

def MRAE(y_pred, y_true):
    return np.sum((np.abs((y_pred - y_true)/y_true)/y_true.shape[1]).reshape(y_pred.shape),axis=1)

def accuracy(y_pred, y_true, metric=True):
    
    #y_pred = y_pred[0]
    y_true = y_true[0]
    
    y_tmp = (np.array(y_pred[0]) > 0.5).astype(int)
    correct = 0
    for i in range(len(y_tmp)):
        if y_tmp[i] == y_true[i]: correct += 1
    
    if metric: return (correct / float(len(y_tmp)) * 100.0)*np.ones(y_pred.shape)/y_pred.shape[1]

    return correct / float(len(y_tmp)) * 100.0

def MSE_metric(y_pred,y_true):
    return ((y_pred - y_true)**2/y_pred.shape[1]).reshape(y_pred.shape).sum(axis=1)
 
lossfunc = {'MSE':(MSE,d_MSE),
            'binary_crossentropy':(binary_crossentropy,d_binary_crossentropy)}

