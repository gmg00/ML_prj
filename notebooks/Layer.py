import numpy as np
from functions import act_func, d_act_func
from Optimizer import Optimizer

class Layer:

    def __init__(self, prev_layer, dim_layer, act_function, init_weights_mode='rand'):
<<<<<<< HEAD
        """ Initialize Layer object.

        Args:
            prev_layer (Layer): previous layer in the network.
            dim_layer (int): number of units in the layer.
            act_function (function): activation function of the layer.
            init_weights_mode (str, optional): mode of weight initialization. Defaults to 'rand'.
        """        
  
        self.prev_layer = prev_layer
        self.init_params(dim_layer, act_function, init_weights_mode) 

    def init_params(self, dim_layer, act_function, init_weights_mode):
        """ Initialize paamters of the Layer object.

        Args:
            dim_layer (int): number of units in the layer.
            act_function (function): activation function of the layer.
            init_weights_mode (str): mode of weight initialization.
        """        
        # Set number of units.
        self.dim_layer = dim_layer
        # Initialize weights and biases.
        self.init_weights(init_weights_mode)

        self.layer = None 
=======
  
        self.prev_layer = prev_layer
        self.init_params(dim_layer, act_function, init_weights_mode) #self.init_params(dim_layer, act_function, input)

    def init_params(self, dim_layer, act_function, init_weights_mode):
        #self.dim_batch = self.prev_layer.dim_batch
        self.dim_layer = dim_layer
        #self.W = np.random.uniform(-0.5, 0.5, (dim_layer, self.prev_layer.dim_layer))    #inizializzo la matrice dei pesi
        #self.b = np.random.uniform(-0.5, 0.5, (dim_layer, 1))      #inizializzo il vettore dei bias
        self.init_weights(init_weights_mode)
        self.layer = None #np.empty((dim_layer, self.dim_batch))
>>>>>>> main
        self.z = None
        self.target = None

        # Vectorize activation function and its derivative.
        self.act_function = np.vectorize(act_func[act_function])
        self.d_act_function = np.vectorize(d_act_func[act_function])

        # Initialize Adam optimizer.
        self.opt = Optimizer(self.dim_layer, self.prev_layer.dim_layer)

        self.eta = 0

        # Initialize velocity variables.
        self.d_W_old = 0
        self.d_b_old = 0
    
    def init_weights(self, mode='rand', range = [-0.5,0.5]):
        """ Initialize weights and biases.

        Args:
            mode (str, optional): if mode is 'rand' initialize weights from a random uniform distribution, if mode is 'xavier' initialize weights
            following Xavier/Golgot approach. Defaults to 'rand'.
            range (list, optional): range for the random uniform distribution. Defaults to [-0.5,0.5].
        """        
        if mode == 'rand':
            self.W = np.random.uniform(range[0], range[1], (self.dim_layer, self.prev_layer.dim_layer))    
            self.b = np.random.uniform(range[0], range[1], (self.dim_layer, 1))
        if mode == 'xavier':
            self.W = np.random.normal(0, 2/self.dim_layer, (self.dim_layer, self.prev_layer.dim_layer))
            self.b = np.random.normal(0, 2/self.dim_layer, (self.dim_layer, 1))

    def get_weights(self, arr=[]):
        """ Get weight matrix and bias.

        Args:
            arr (list, optional): list containing weight matrices and biases of previous layers. Defaults to [].

        Returns:
            list: list containing weight matrices and biases of current and previous layers.
        """        

        arr = self.prev_layer.get_weights(arr)
        arr.append([self.W,self.b])
        return arr
    
    def set_weights(self,arr,i=0):
        """ Set weight matrix and bias for the layer.

        Args:
            arr (list): list of weight matrices and biases for every layer in the network.
            i (int, optional): index for extracting the right weight matrix and bias for the layer. Defaults to 0.

        Returns:
            int: index for the next layer.
        """        
        i = self.prev_layer.set_weights(arr,i)
        self.W = arr[i][0]
        self.b = arr[i][1]
        return i+1
    
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
<<<<<<< HEAD
        """ Compute forward propagation ricursively.

        Args:
            mode (str, optional): if mode is 'train' modify net and layer matrices, if mode is 'predict' leave them unchanged. Defaults to 'train'.

        Returns:
            np.array: layer matrix.
=======
        """ Compute forward propagation for the current layer recursively.

        Args:
            mode (str, optional): if 'train' modify net and layer, if 'predict' leave them unchanged. Defaults to 'train'.

        Returns:
            np.array: current layer array.
>>>>>>> main
        """        
    
        if mode == 'train':
            self.z = self.W.dot(self.prev_layer.forward()) + self.b
            self.layer = self.act_function(self.z)

            return self.layer
        elif mode == 'predict':
            # Compute layer matrix without updating its and net values.
            return self.act_function(self.W.dot(self.prev_layer.forward(mode = 'predict')) + self.b)
        
    def backward(self, next_delta = None, next_weights = None, lossfunc = None, last = False):
        """ Compute backward propagation ricursively.

        Args:
            next_delta (np.array, optional): delta of the next layer. Defaults to None.
            next_weights (np.array, optional): weights of the next layer. Defaults to None.
            lossfunc (func, optional): loss function of the network. Defaults to None.
            last (bool, optional): if True perform backward for output layer, otherwise perform backward for hidden layer. Defaults to False.

        Returns:
            np.array: layer array.
        """    
        if last == True:
            delta = self.d_act_function(self.z) * lossfunc(self.layer,self.target) 
           
        else:
            delta = self.d_act_function(self.z) * next_weights.T.dot(next_delta)

        self.d_W = delta.dot(self.prev_layer.backward(delta,self.W).T)
        self.d_b = delta.sum(axis=1).reshape((delta.shape[0],1))
        return self.layer

    def update_weights(self, eta, lam = 0., alpha = 0., l1_reg = False, use_opt = 0, nest=False):
        """ Update weight matrix and biases.

        Args:
            eta (float): learning rate.
            lam (float, optional): regularization parameter. Defaults to 0.
            alpha (float, optional): momentum parameter. Defaults to 0.
            l1_reg (bool, optional): if True use Lasso regularization, otherwise use Ridge regularization. Defaults to False.
            use_opt (int, optional): if 1 use Adam optimizer. Defaults to 0.
            nest (bool, optional): if True use Nesterov momentum approach, otherwise use normal momentum. Defaults to False.
        """        

<<<<<<< HEAD
        if nest: 
            self.W = self.W_old
            self.b = self.b_old
=======
            self.d_W = delta.dot(self.prev_layer.backward_nest(delta,self.W_projected).T)
            self.d_b = delta.sum(axis=1).reshape((delta.shape[0],1))
            return self.layer_projected

    def update_weights(self, eta, lam, alpha, l1_reg = False, use_opt = 0, nest=False):
>>>>>>> main

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

<<<<<<< HEAD
        self.prev_layer.update_weights(eta, lam, alpha, l1_reg, use_opt, nest)

    def nest_update(self, alpha):
        """ Update weights following in Nesterov approach.
=======
        self.prev_layer.update_weights(eta, lam, alpha, l1_reg, use_opt)

    def nest_update(self, alpha):
        """ Update weights before computing gradient as in Nestorov approach.
>>>>>>> main

        Args:
            alpha (float): momentum parameter.
        """        
<<<<<<< HEAD
        self.W_old = self.W
        self.b_old = self.b
        self.W = self.W + alpha * self.d_W_old
        self.b = self.b + alpha * self.d_b_old
=======
        self.W_projected = self.W + alpha * self.d_W_old
        self.b_projected = self.b + alpha * self.d_b_old
>>>>>>> main
        self.prev_layer.nest_update(alpha)

    def reset_velocity(self):
        """ Reset velocity of the layer.
        """        
        self.d_W_old = 0
        self.d_b_old = 0
        self.prev_layer.reset_velocity()
    
class Input(Layer):

    def __init__(self, input_dim):
        """ Initialize Input layer object.

        Args:
            input_dim (int): number of input layer units.
        """        

        Layer.__init__(self, None, input_dim, 'lin')

    def init_params(self, dim_layer, act_function, init_weights_mode):
<<<<<<< HEAD
        """ Initialize input layer parameters.

        Args:
            dim_layer (int): number of input layer units.
            act_function (func): not relevant for input layer.
            init_weights_mode (str): not relevant for input layer.
        """        
=======
>>>>>>> main
        self.layer = None

        self.dim_layer = dim_layer

    def forward(self, mode = 'train'):
        """ Perform forward for input layer.

        Args:
            mode (str, optional): not relevant for input layer. Defaults to 'train'.

        Returns:
            np.array: input array.
        """        
        return self.layer
    
    def backward(self, next_delta = None, next_weights = None):
        """ Perform backward for input layer.

        Args:
            next_delta (np.array, optional): not relevant for input layer. Defaults to None.
            next_weights (np.array, optional): not relevant for input layer. Defaults to None.

        Returns:
            np.array: input array.
        """        
        
        return self.layer
    
<<<<<<< HEAD
    def update_weights(self, eta, lam=0, alpha=0,l1_reg=False,use_opt=0,nest=False):
        """ Not relevant for input layer.
        """        
=======
    def update_weights(self, eta, lam=0, alpha=0,l1_reg=False,use_opt=0):
>>>>>>> main
        pass
    
    def reset_velocity(self):
        """ Not relevant for input layer.
        """     
        pass

    def nest_update(self, alpha):
        """ Not relevant for input layer.
        """     
        pass

    def get_weights(self, weights_dict, layer='hidden'):
        """ Not relevant for input layer.
        """             
        return []
        
    def set_weights(self, arr, i=0):
        """ Not relevant for input layer.
        """     
        return 0

<<<<<<< HEAD
        
=======
        return self.layer
    
    def get_initial_weights(self, weights_dict, layer='hidden'):
        return []
        
    def set_weights(self, arr, i=0):
        return 0
>>>>>>> main
