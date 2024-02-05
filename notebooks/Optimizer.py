import numpy as np

class Optimizer:

    def __init__(self, dim_layer, dim_prev_layer, eta=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        """ Initialize Optimizer object.

        Args:
            dim_layer (int): number of units.
            dim_prev_layer (Layer): previous layer.
            eta (float, optional): learning rate. Defaults to 0.01.
            beta1 (float, optional): parameter beta1. Defaults to 0.9.
            beta2 (float, optional): parameter beta2. Defaults to 0.999.
            eps (float, optional): parameter usefull to avoid dividing by zero. Defaults to 1e-8.
        """        
        self.m_dw = np.zeros((dim_layer, dim_prev_layer))
        self.v_dw = np.zeros((dim_layer, dim_prev_layer))
        self.m_db = np.zeros((dim_layer,1))
        self.v_db = np.zeros((dim_layer,1))
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.eta = eta
        self.t = 1
        self.dw_old = 0
        self.db_old = 0

    def update_eta(self,eta):
        """ Update eta.

        Args:
            eta (float): learning rate.
        """        
        self.eta = eta

    def update(self, w, b, dw, db):
        """ Update weights.

        Args:
            w (np.array): weight matrix.
            b (np.array): bias.
            dw (np.array): delta of weights.
            db (_type_): delta of biases.

        Returns:
            np.array, np.array: updated weights, updated bias.
        """        

        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw

        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db

        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw**2)

        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * (db**2)

        #bias correction
        m_dw_corr = self.m_dw / (1 - self.beta1**self.t)
        m_db_corr = self.m_db / (1 - self.beta1**self.t)
        v_dw_corr = self.v_dw / (1 - self.beta2**self.t)
        v_db_corr = self.v_db / (1 - self.beta2**self.t)

        #update weights and biases
        w = w - self.eta * (m_dw_corr / (np.sqrt(v_dw_corr) + self.eps))
        b = b - self.eta * (m_db_corr / (np.sqrt(v_db_corr) + self.eps))
        
        self.t += 1

        return w, b
