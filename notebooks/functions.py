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
'''def relu(x):
    if x > 0: return x
    else: return 0

def d_relu(x):
    if x > 0: return 1
    else: return 0'''

def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function.

    Parameters:
    - x: Input array.

    Returns:
    - Output of the ReLU activation applied element-wise to the input.
    """
    return np.maximum(0, x)

def d_relu(x):
    """
    Derivative of the Rectified Linear Unit (ReLU) activation function.

    Parameters:
    - x: Input array.

    Returns:
    - Output of the derivative of the ReLU activation applied element-wise to the input.
    """
    return np.where(x > 0, 1, 0)
    

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

'''def MSE(layer, target):
    #print(layer.shape)
    return ((layer - target)**2/target.shape[1]).reshape(layer.shape)'''

def MSE(y_pred, y_true):
    y_pred = y_pred.reshape(y_true.shape)
    return np.mean((y_pred - y_true)**2)

'''def MEE(layer, target):
    return (np.sqrt(np.sum((layer - target)**2, axis=0).reshape((1,layer.shape[1]))))/layer.shape[1]'''

def MEE(y_pred, y_true):
    y_pred = y_pred.reshape(y_true.shape)
    return np.mean(np.sqrt(np.sum((y_pred - y_true)**2, axis=0)))

def binary_crossentropy(layer, target, epsilon=1e-15):
    layer = np.clip(layer, epsilon, 1 - epsilon)  # Avoid division by zero
    return np.mean(- (target * np.log(layer) + (1 - target) * np.log(1 - layer)))

def d_binary_crossentropy(layer, target, epsilon=1e-15):
    layer = np.clip(layer, epsilon, 1 - epsilon)  # Avoid division by zero
    return (layer - target)/(layer*(1-layer)*target.shape[1])

def MRAE(y_pred, y_true):
    return np.sum((np.abs((y_pred - y_true)/y_true)/y_true.shape[1]).reshape(y_pred.shape),axis=1)

def accuracy(y_pred, y_true):
    
    y_true = y_true[0]
    
    y_tmp = (np.array(y_pred[0]) > 0.5).astype(int)
    correct = 0
    for i in range(len(y_tmp)):
        if y_tmp[i] == y_true[i]: correct += 1

    return correct / float(len(y_tmp)) * 100.0

 
lossfunc = {'MSE':(MSE,d_MSE),
            'binary_crossentropy':(binary_crossentropy,d_binary_crossentropy)}

