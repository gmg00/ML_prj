import numpy as np
from functions import act_func, d_act_func
from Optimizer import Optimizer

class Layer:

    def __init__(self, prev_layer, dim_layer, act_function):
  
        self.prev_layer = prev_layer
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
        self.opt = Optimizer(self.dim_layer, self.prev_layer.dim_layer)
        self.eta = 0
        self.d_W_old = 0
        self.d_b_old = 0
    

    def forward(self, mode = 'train'):
        if mode == 'train':
            self.z = self.W.dot(self.prev_layer.forward()) + self.b
            self.layer = self.act_function(self.z)

            return self.layer
        elif mode == 'predict':
            return self.act_function(self.W.dot(self.prev_layer.forward(mode = 'predict')) + self.b)
        
    
    def backward(self, next_delta = None, next_weights = None, lossfunc = None, last = False):
        
        if last == True:

            delta = self.d_act_function(self.z) * lossfunc(self.layer,self.target)

            self.d_W = delta.dot(self.prev_layer.backward(delta,self.W).T)
            self.d_b = delta.sum(axis=1).reshape((delta.shape[0],1))
            return self.layer 
           
        else:

            delta = self.d_act_function(self.z) * next_weights.T.dot(next_delta)

            self.d_W = delta.dot(self.prev_layer.backward(delta,self.W).T)
            self.d_b = delta.sum(axis=1).reshape((delta.shape[0],1))
            return self.layer

    def update_weights(self, eta, lam = 0, alpha = 0, l1_reg = False, use_opt = 1):

        if self.eta != eta:
            self.eta = eta
            self.opt.update_eta(self.eta)

        if use_opt == 1:
            self.d_W = self.d_W + lam * self.W
            self.W, self.b = self.opt.update(self.W, self.b, self.d_W, self.d_b)
        else:
            if l1_reg: reg = + lam * np.sign(self.W)
            else: reg = + lam * self.W
            self.d_W_old = alpha * self.d_W_old - eta * self.d_W - reg
            self.d_b_old = alpha * self.d_b_old - eta * self.d_b
            self.W = self.W + self.d_W_old
            self.b = self.b + self.d_b_old

        self.prev_layer.update_weights(eta, lam, alpha, use_opt)

    def reset_velocity(self):
        self.d_W_old = 0
        self.d_b_old = 0
        self.prev_layer.reset_velocity()
    
class Input(Layer):

    def __init__(self, input_dim):

        Layer.__init__(self, None, input_dim, 'lin')

    def init_params(self, dim_layer, act_function):
        self.layer = None

        self.dim_layer = dim_layer

    def forward(self, mode = 'train'):

        return self.layer
    
    def backward(self, next_delta = None, next_weights = None):
        
        return self.layer
    
    def update_weights(self, eta, lam=0, alpha=0,use_opt=1):
        pass
    
    def reset_velocity(self):
        pass
        