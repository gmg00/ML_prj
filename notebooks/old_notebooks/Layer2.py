import numpy as np
from matplotlib import pyplot as plt
import torch
import pandas as pd
import math





#linear activation function 
def linear(x):
    """_summary_

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
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """    
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """    
    f =sigmoid(x)
    return f * (1-f)

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
    return (layer - target)**2/target.shape[1]

def binary_crossentropy(layer, target):
    return - (target * np.log(layer) + (1 - target) * np.log(1 - layer))/target.shape[1]

def d_binary_crossentropy(layer, target):
    return (layer - target)/(layer*(1-layer)*target.shape[1])

class Layer:

    def __init__(self, prev_layer, dim_layer, act_function):
  
        self.prev_layer = prev_layer
        #self.next_layer = None
        #self.target = target
        self.init_params(dim_layer, act_function) #self.init_params(dim_layer, act_function, input)

    def init_params(self, dim_layer, act_function):
        #self.dim_batch = self.prev_layer.dim_batch
        self.dim_layer = dim_layer
        self.W = np.random.uniform(-0.5, 0.5, (dim_layer, self.prev_layer.dim_layer))    #inizializzo la matrice dei pesi
        self.b = np.random.uniform(-0.5, 0.5, (dim_layer, 1))      #inizializzo il vettore dei bias
        self.layer = None #np.empty((dim_layer, self.dim_batch))
        self.z = None
        self.target = None
        self.act_function = np.vectorize(act_func[act_function])
        self.d_act_function = np.vectorize(d_act_func[act_function])
        

    def forward(self, mode = 'train'):
        if mode == 'train':
            self.z = self.W.dot(self.prev_layer.forward()) + self.b
            self.layer = self.act_function(self.z)
            print(f'layer = {self.layer.shape}')
            return self.layer
        elif mode == 'predict':
            return self.act_function(self.W.dot(self.prev_layer.forward(mode = 'predict')) + self.b)
        
    
    def backward(self, next_delta = None, next_weights = None, lossfunc = None, last = False):
        #print(f'Entered backward: target = {self.target}')
        
        if last == True:

            #print('self.target != None: output')
            #print(self.d_act_function(self.z).shape)
            #print(lossfunc(self.layer,self.target).shape)
            #print(self.layer.shape)
            #print(self.target.shape)
            delta = self.d_act_function(self.z) * lossfunc(self.layer,self.target)
            #self.prev_layer.backward(delta,self.weights)
            self.d_W = delta.dot(self.prev_layer.backward(delta,self.W).T)
            self.d_b = delta.sum(axis=1).reshape((delta.shape[0],1))
            return self.layer 
           
        else:
            #print('self.target == None: hidden')
            delta = self.d_act_function(self.z) * next_weights.T.dot(next_delta)
            #self.prev_layer.backward(delta,self.weights)
            self.d_W = delta.dot(self.prev_layer.backward(delta,self.W).T)
            self.d_b = delta.sum(axis=1).reshape((delta.shape[0],1))
            return self.layer

    def update_weights(self, eta, lam):

        self.W = self.W - eta * self.d_W - lam * self.W
        self.prev_layer.update_weights(eta, lam)

    def err(self):
        return np.sqrt((self.layer[0]-self.target[0])**2+(self.layer[1]-self.target[1])**2+(self.layer[2]-self.target[2])**2).mean()
    
    def rel_err(self):
        return np.sqrt((self.layer[0]-self.target[0])**2/self.target[0]**2+(self.layer[1]-self.target[1])**2/self.target[1]**2+(self.layer[2]-self.target[2])**2/self.target[2]**2).mean()

    def err_i(self,i):
        return np.sqrt((self.layer[i]-self.target[i])**2).mean()
    
    def rel_err_i(self,i):
        return (np.sqrt((self.layer[i]-self.target[i])**2)/self.target[i]).mean()
    
class Input(Layer):
    static_var = 0
    def __init__(self, input_dim):
        Input.static_var += 1
        self.instance_var = Input.static_var
        Layer.__init__(self, None, input_dim, 'lin')

    def init_params(self, dim_layer, act_function):
        self.layer = None
        #self.dim_batch = input.shape[1]
        self.dim_layer = dim_layer

    def forward(self, mode = 'train'):
        #print(self.layer.shape)
        return self.layer
    
    def backward(self, next_delta = None, next_weights = None):
        return self.layer
    
    def update_weights(self, eta, lam):
        return self.layer
    
    def prova(self):
        return self.dim_layer
        
lossfunc = {'MSE':(MSE,d_MSE),
            'binary_crossentropy':(binary_crossentropy,d_binary_crossentropy)}
        
if __name__ == '__main__':

    names = ['id', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 
         'feature_7', 'feature_8', 'feature_9', 'feature_10', 'target_x', 'target_y','target_z']

    df = pd.read_csv("/mnt/c/Users/HP/Desktop/UNI/LM_1/MachineLearning/ML_prj/data/ML-CUP23-TR.csv", names=names, comment='#')

    targets = ['target_x', 'target_y', 'target_z']
    features = list(set(names) - {'id', 'target_x', 'target_y', 'target_z'})
    prova = df[0:1000]
    X_train, y = prova[features].to_numpy().T, prova[targets].to_numpy().T

    eta = 0.01
    lam = 0.00
    o = 0
    E1 = []
    E2 = []
    E3 = []
    E = []
    dim_batch = 100
    input_layer = Input(10)
    hidden_layer = Layer(input_layer, 8, 'relu')
    output_layer = Layer(hidden_layer, 3, 'lin')
    while o<1000:
        err1 = []
        err2 = []
        err3 = []
        input_layer.layer = X_train
        output_layer.target = y
        output_layer.forward()
        err1.append(output_layer.rel_err_i(0))
        err2.append(output_layer.rel_err_i(1))
        err3.append(output_layer.rel_err_i(2))
        print(output_layer.layer)
        print(output_layer.target)
        output_layer.backward(lossfunc = d_binary_crossentropy, last=True)
        output_layer.update_weights(eta, lam)
        if o % 100 == 0: print(o)

        E1.append(np.array(err1).mean())
        E2.append(np.array(err2).mean())
        E3.append(np.array(err3).mean())
        E.append(output_layer.err())
        o += 1

    print(np.array(E).shape)
    plt.plot(E)
    plt.show()