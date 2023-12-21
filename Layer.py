import numpy as np
from Activation functions import *

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

