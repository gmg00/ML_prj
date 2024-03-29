import numpy as np
from functions import act_func, d_act_func
from Optimizer import Optimizer

class Layer:

    def __init__(self, prev_layer, dim_layer, act_function, init_weights_mode='rand'):
  
        self.prev_layer = prev_layer
        self.init_params(dim_layer, act_function, init_weights_mode) #self.init_params(dim_layer, act_function, input)

    def init_params(self, dim_layer, act_function, init_weights_mode):
        #self.dim_batch = self.prev_layer.dim_batch
        self.dim_layer = dim_layer
        #self.W = np.random.uniform(-0.5, 0.5, (dim_layer, self.prev_layer.dim_layer))    #inizializzo la matrice dei pesi
        #self.b = np.random.uniform(-0.5, 0.5, (dim_layer, 1))      #inizializzo il vettore dei bias
        self.init_weights(init_weights_mode)
        self.layer = None #np.empty((dim_layer, self.dim_batch))
        self.z = None
        self.target = None
        self.act_function = np.vectorize(act_func[act_function])
        self.d_act_function = np.vectorize(d_act_func[act_function])
        self.opt = Optimizer(self.dim_layer, self.prev_layer.dim_layer)
        self.eta = 0
        self.d_W_old = 0
        self.d_b_old = 0

        self.W_projected = 0
        self.b_projected  = 0
        self.z_projected = 0
        self.layer_projected = 0
    
    def init_weights(self, mode='rand', range = [-0.5,0.5]):
        """ Initialize weight matrix and biases for the layer.

        Args:
            mode (str, optional): if 'rand' compute weights with a random uniform distribution between limits of range,
            if 'xavier' use Xavier/Glorot weight initialization. Defaults to 'rand'.
            range (list, optional): range for the uniform distribution. Defaults to [-0.5,0.5].
        """        
        if mode == 'rand':
            self.W = np.random.uniform(range[0], range[1], (self.dim_layer, self.prev_layer.dim_layer))    
            self.b = np.random.uniform(range[0], range[1], (self.dim_layer, 1))
        if mode == 'xavier':
            self.W = np.random.normal(0, 2/self.dim_layer, (self.dim_layer, self.prev_layer.dim_layer))
            self.b = np.random.normal(0, 2/self.dim_layer, (self.dim_layer, 1))

    def get_initial_weights(self, arr=[]):
        
        arr = self.prev_layer.get_initial_weights(arr)
        arr.append([self.W,self.b])
        return arr
    
    def set_weights(self,arr,i=0):
        """ Set weights and biases for the layer.

        Args:
            arr (list): list of every weight and bias matrix in the network.
            i (int, optional): layer index for extraction of the right matrices from the list. Defaults to 0.

        Returns:
            int: index for the next layer.
        """        
        i = self.prev_layer.set_weights(arr,i)
        self.W = arr[i][0]
        self.b = arr[i][1]
        return i+1

    def forward(self, mode = 'train'):
        """ Compute forward propagation for the current layer recursively.

        Args:
            mode (str, optional): if 'train' modify net and layer, if 'predict' leave them unchanged. Defaults to 'train'.

        Returns:
            np.array: current layer array.
        """        
    
        if mode == 'train':
            self.z = self.W.dot(self.prev_layer.forward()) + self.b
            self.layer = self.act_function(self.z)

            return self.layer
        elif mode == 'predict':
            return self.act_function(self.W.dot(self.prev_layer.forward(mode = 'predict')) + self.b)
        
        elif mode == 'nest':
            self.z_projected = self.W_projected.dot(self.prev_layer.forward(mode = 'nest')) + self.b_projected
            self.layer_projected = self.act_function(self.z_projected)
            return self.layer_projected
        
    
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
        
    def backward_nest(self, next_delta = None, next_weights = None, lossfunc = None, last = False):
    
        if last == True:

            delta = self.d_act_function(self.z_projected) * lossfunc(self.layer_projected,self.target)
            self.d_W = delta.dot(self.prev_layer.backward_nest(delta,self.W_projected).T)
            self.d_b = delta.sum(axis=1).reshape((delta.shape[0],1))
            return self.layer_projected
           
        else:

            delta = self.d_act_function(self.z_projected) * next_weights.T.dot(next_delta)

            self.d_W = delta.dot(self.prev_layer.backward_nest(delta,self.W_projected).T)
            self.d_b = delta.sum(axis=1).reshape((delta.shape[0],1))
            return self.layer_projected

    def update_weights(self, eta, lam, alpha, l1_reg = False, use_opt = 0, nest=False):

        if self.eta != eta:
            self.eta = eta
            self.opt.update_eta(self.eta)

        if use_opt == 1:
            if l1_reg: reg = + lam * np.sign(self.W)
            else: reg = + lam * self.W
            self.d_W = self.d_W + reg
            self.W, self.b = self.opt.update(self.W, self.b, self.d_W, self.d_b)
        else:
            if l1_reg: reg = + lam * np.sign(self.W)
            else: reg = + lam * self.W
            self.d_W_old = alpha * self.d_W_old - eta * self.d_W - reg
            self.d_b_old = alpha * self.d_b_old - eta * self.d_b
            self.W = self.W + self.d_W_old
            self.b = self.b + self.d_b_old

        self.prev_layer.update_weights(eta, lam, alpha, l1_reg, use_opt)

    def nest_update(self, alpha):
        """ Update weights before computing gradient as in Nestorov approach.

        Args:
            alpha (float): momentum parameter.
        """        
        self.W_projected = self.W + alpha * self.d_W_old
        self.b_projected = self.b + alpha * self.d_b_old
        self.prev_layer.nest_update(alpha)

    def reset_velocity(self):
        self.d_W_old = 0
        self.d_b_old = 0
        self.prev_layer.reset_velocity()
    
class Input(Layer):

    def __init__(self, input_dim):

        Layer.__init__(self, None, input_dim, 'lin')

    def init_params(self, dim_layer, act_function, init_weights_mode):
        self.layer = None

        self.dim_layer = dim_layer

    def forward(self, mode = 'train'):
        return self.layer
    
    def backward(self, next_delta = None, next_weights = None):
        
        return self.layer
    
    def update_weights(self, eta, lam=0, alpha=0,l1_reg=False,use_opt=0):
        pass
    
    def reset_velocity(self):
        pass

    def nest_update(self, alpha):
        pass

    def backward_nest(self, next_delta = None, next_weights = None):

        return self.layer
    
    def get_initial_weights(self, weights_dict, layer='hidden'):
        return []
        
    def set_weights(self, arr, i=0):
        return 0