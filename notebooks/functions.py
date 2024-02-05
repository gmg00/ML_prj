import numpy as np
import math

#linear activation function 
def linear(x):
    """ Linear function.

    Args:
        x (np.array): array.

    Returns:
        np.array: new array.
    """    
    return x

def d_linear(x):
    """ Derivative of linear function.

    Args:
        x (np.array): array.

    Returns:
        np.array: new array.
    """    
    return 1


def sigmoid(x):
    """ Sigmoid function.

    Args:
        x (np.array): array.

    Returns:
        np.array: new array.
    """    
    return np.exp(-np.logaddexp(0, -x))

def d_sigmoid(x):
    """ Derivative of sigmoid function.

    Args:
        x (np.array): array.

    Returns:
        np.array: new array.
    """   
    f = sigmoid(x)
    return f * (1 - f)


def relu(x):
    """ Rectified Linear Unit (ReLU) activation function.

    Args:
        x (np.array): array.

    Returns:
        np.array: new array.
    """
    return np.maximum(0, x)

def d_relu(x):
    """ Derivative rectified Linear Unit (ReLU) activation function.

    Args:
        x (np.array): array.

    Returns:
        np.array: new array.
    """
    return np.where(x > 0, 1, 0)
    
def leaky_relu(x, alpha=0.01):
    """ Leaky Relu activation function.

    Args:
        x (np.array): array.
        alpha (float): parameter for negative xs.

    Returns:
        np.array: new array.
    """
    return np.where(x > 0, x, alpha * x)

def d_leaky_relu(x, alpha=0.01):
    """ Derivative of leaky Relu activation function.

    Args:
        x (np.array): array.
        alpha (float): parameter for negative xs.

    Returns:
        np.array: new array.
    """
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    """ Exponential Relu activation function.

    Args:
        x (np.array): array.
        alpha (float): parameter for negative xs.

    Returns:
        np.array: new array.
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def d_elu(x, alpha=1.0):
    """ Derivative of exponential Relu activation function.

    Args:
        x (np.array): array.
        alpha (float): parameter for negative xs.

    Returns:
        np.array: new array.
    """
    return np.where(x > 0, 1, alpha * np.exp(x))

def TanH(x):
    """ Iperbolic tangent function.

    Args:
        x (np.array): array.

    Returns:
        np.array: new array.
    """
    return np.tanh(x)

def d_TanH(x):
    """ Iperbolic tangent function.

    Args:
        x (np.array): array.

    Returns:
        np.array: new array.
    """
    return 1 - np.tanh(x)**2

#Dictionary for the activation functions
act_func = {
    'lin': linear,
    'sigm': sigmoid,
    'relu': relu,
    'tanh': TanH,
    'leaky_relu' : leaky_relu,
    'elu' : elu
}

#A second dictionary for their derivatives
d_act_func = {
    'lin': d_linear,
    'sigm': d_sigmoid,
    'relu': d_relu,
    'tanh': d_TanH,
    'leaky_relu' : d_leaky_relu,
    'elu' : d_elu
}

def d_MSE(layer, target):
    """ Derivative of mean squared error.

    Args:
        layer (np.array): prediction array.
        target (np.array): true values array.

    Returns:
        np.array: new array.
    """    
    return 2*(layer - target)/target.shape[1]


def MSE(y_pred, y_true):
    """ Mean squared error.

    Args:
        layer (np.array): prediction array.
        target (np.array): true values array.

    Returns:
        float: value of MEE.
    """    
    y_pred = y_pred.reshape(y_true.shape)
    return np.mean((y_pred - y_true)**2)

def MEE(y_pred, y_true):
    """ Mean euclidean error.

    Args:
        layer (np.array): prediction array.
        target (np.array): true values array.

    Returns:
        float: value of MEE.
    """   
    y_pred = y_pred.reshape(y_true.shape)
    return np.mean(np.sqrt(np.sum((y_pred - y_true)**2, axis=0)))

def binary_crossentropy(layer, target, epsilon=1e-15):
    """ Binary cross entropy.

    Args:
        layer (np.array): prediction array.
        target (np.array): true values array.
        epsilon (float, optional): parameter to avoid division by 0. Defaults to 1e-15.

    Returns:
        float: value of binary crossentropy.
    """    
    layer = np.clip(layer, epsilon, 1 - epsilon)  # Avoid division by zero
    return np.mean(- (target * np.log(layer) + (1 - target) * np.log(1 - layer)))

def d_binary_crossentropy(layer, target, epsilon=1e-15):
    """ Derivative of binary cross entropy.

    Args:
        layer (np.array): prediction array.
        target (np.array): true values array.
        epsilon (float, optional): parameter to avoid division by 0. Defaults to 1e-15.

    Returns:
        np.array: new array.
    """       
    layer = np.clip(layer, epsilon, 1 - epsilon)  # Avoid division by zero
    return (layer - target)/(layer*(1-layer)*target.shape[1])

def MRAE(y_pred, y_true):
    """ Mean relative absolute error.

    Args:
        layer (np.array): prediction array.
        target (np.array): true values array.

    Returns:
        float: value of MRAE.
    """   
    return np.sum((np.abs((y_pred - y_true)/y_true)/y_true.shape[1]).reshape(y_pred.shape),axis=1)

def accuracy(y_pred, y_true):
    """ Accuracy.

    Args:
        layer (np.array): prediction array.
        target (np.array): true values array.

    Returns:
        float: value of accuracy.
    """       
    y_true = y_true[0]
    
    y_tmp = (np.array(y_pred[0]) > 0.5).astype(int)
    correct = 0
    for i in range(len(y_tmp)):
        if y_tmp[i] == y_true[i]: correct += 1

    return correct / float(len(y_tmp)) * 100.0

# Dictionary with loss functions
lossfunc = {'MSE':(MSE,d_MSE),
            'binary_crossentropy':(binary_crossentropy,d_binary_crossentropy)}

