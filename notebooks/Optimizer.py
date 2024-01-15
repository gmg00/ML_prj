import numpy as np

class Optimizer:

    def __init__(self, dim_layer, dim_prev_layer, eta=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m_dw = np.zeros((dim_layer, dim_prev_layer))
        self.v_dw = np.zeros((dim_layer, dim_prev_layer))
        self.m_db = np.zeros((dim_layer,1))
        self.v_db = np.zeros((dim_layer,1))
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.eta = eta
        self.t = 1

    def update_eta(self,eta):
        self.eta = eta

    def update(self, w, b, dw, db, lam = 0):

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
        w = w - self.eta * (m_dw_corr / (np.sqrt(v_dw_corr) + self.eps)) - lam * w
        b = b - self.eta * (m_db_corr / (np.sqrt(v_db_corr) + self.eps)) - lam * b


        self.t += 1

        return w, b
