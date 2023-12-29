import numpy as np
import math


#linear activation function 
def linear(x):
    return x

def d_linear(x):
    return 1

#sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    f =sigmoid(x)
    return f * (1-f)

#ReLu activation function
def relu(x):
    return np.max(0,x)

def d_relu(x):
    return int(x>0)

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

class Layer:
    def __init__(self, n_in, n_out, prev_layer, next_layer, act_func, batch_norm):
        self.n_in = n_in
        self.n_out = n_out
        self.W = np.random.uniform(-0.2, 0.2, (n_out, n_in))    #inizializzo la matrice dei pesi
        self.b = np.random.uniform(-0.2, 0.2, (n_out))      #inizializzo il vettore dei bias
        self.act_func = np.vectorize(func_dict[act_func])
        self.d_act_func = np.vectorize(d_act_func[act_func])
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.batch_norm = batch_norm    #batch_norm sarà True/False

    def forward(self, x_batch): #x_batch deve avere dimensione (n_in x n_batch)
        if self.prev_layer == None :
            Y = self.act_func(self.W @ x_batch + self.b)
        else:
            Y = self.act_func(self.W @ self.prev_layer.forward(x_batch) + self.b)
        if self.batch_norm:
            self.mu = np.mean(Y, axis=1)
            self.sigma = np.std(Y, axis=1)
            Y = (Y - self.mu)/self.sigma    #standardizzo il batch
        return Y

class Layer:

    def __init__(self, input, prev_layer, dim_layer, act_function, dim_batch, target):

        self.dim_layer = dim_layer
        self.prev_layer = prev_layer
        self.act_function = act_func[act_function]
        self.d_act_function = d_act_func[act_function]
        self.dim_batch = dim_batch
        self.target = target

        if self.prev_layer != None:
            self.W = np.random.uniform(-0.2, 0.2, (self.dim_layer, self.prev_layer.dim_layer))    #inizializzo la matrice dei pesi
            self.b = np.random.uniform(-0.2, 0.2, (self.dim_layer, self.dim_batch))      #inizializzo il vettore dei bias
            self.layer = np.empty((self.dim_layer, self.dim_batch))
        else: 
            self.layer = input

    def forward(self):
        if self.prev_layer == None:
            print('prev_layer == None: return self.layer')
            return self.layer
        else: 
            print('prev_layer =! None: compute forward')
            self.z = self.W @ self.prev_layer.forward() + self.b
            self.layer = self.act_function(self.z)
            return self.layer
    
    def backward(self, next_delta, next_weights):
        print(f'Entered backward: target = {self.target}')
        
        if self.target is None:
            if self.prev_layer != None:
                print('self.target == None: hidden')
                delta = next_weights.T @ next_delta
                self.d_W = delta @ self.prev_layer.backward(delta,self.W).T
                self.d_b = delta.sum(axis=1).reshape((delta.shape[0],1))
                return self.layer

            else: 
                print('input')
                return self.layer
            
        else:
            print('self.target != None: output')
            delta = 2 * self.d_act_function(self.z) * (self.layer - self.target)/self.target.shape[1]
            self.d_W = delta @ self.prev_layer.backward(delta,self.W).T
            self.d_b = delta.sum(axis=1).reshape((delta.shape[0],1))
            return self.layer        
