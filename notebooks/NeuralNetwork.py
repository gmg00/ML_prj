import numpy as np
from functions import lossfunc
import sys

class NeuralNetwork:

    def __init__(self, input_layer, output_layer, loss, metrics = []):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.lossfunc = lossfunc[loss][0]
        self.d_lossfunc = lossfunc[loss][1]
        self.metrics = metrics

    def get_initial_weights_list(self):
        return self.output_layer.get_initial_weights()
    
    def set_initial_weights(self,weights_list):
        self.output_layer.set_weights(weights_list)

    def predict(self,input):
        self.input_layer.layer = input
        return self.output_layer.forward(mode = 'predict')
    
    def update_history_batch(self, history, y_pred, y_true, train_or_val):

        history[f'{train_or_val}_loss_var'].append(self.lossfunc(y_pred, y_true))
        for m in self.metrics:
            history[f'{train_or_val}_{m.__name__}_var'].append(m(y_pred, y_true))

    def update_history_epoch(self, history):
        
        history[f'train_loss'].append(np.mean(history[f'train_loss_var']))
        history[f'val_loss'].append(np.mean(history[f'val_loss_var']))
        history[f'train_loss_var'] = []
        history[f'val_loss_var'] = []
        for m in self.metrics:
            history[f'train_{m.__name__}'].append(np.mean(history[f'train_{m.__name__}_var']))
            history[f'val_{m.__name__}'].append(np.mean(history[f'val_{m.__name__}_var']))
            history[f'train_{m.__name__}_var'] = []
            history[f'val_{m.__name__}_var'] = []

    def print_epoch(self, j, history, eta, val_or_test='val'):
        
        print(f'Epoch {j}: ', end='')
        print(f'train_loss: {history["train_loss"][-1]:.3e}, ', end = '')
        print(f'{val_or_test}_loss: {history[f"{val_or_test}_loss"][-1]:.3e}; ', end = '')
        for m in self.metrics:
            print(f'train_{m.__name__}: {history[f"train_{m.__name__}"][-1]:.3e}, ', end = '')
            print(f'{val_or_test}_{m.__name__}: {history[f"{val_or_test}_{m.__name__}"][-1]:.3e}  ', end = '')
        print(f'lr : {eta}')

    def clear_history(self, history):
    
        del history['train_loss_var']
        del history['val_loss_var']
        for m in self.metrics:
            del history[f'train_{m.__name__}_var']
            del history[f'val_{m.__name__}_var']
        return history

    def check_early_stopping(self, history, early_stopping):

        if early_stopping['compare_function'](history[early_stopping['monitor']][-1], history[early_stopping['monitor']][-2]):
            early_stopping['index'] +=1 
        
        else:
            early_stopping['index'] = 0

        if early_stopping['index'] == early_stopping['patience']: 
            if early_stopping['verbose'] == 1:
                print('Early stopped')
            history = self.clear_history(history)
            return 1,history
        return 0,history
    
    def check_reduce_eta(self, history, reduce_eta, eta):
        if reduce_eta['compare_function'](history[reduce_eta['monitor']][-1], history[reduce_eta['monitor']][-2]):
            reduce_eta['index'] +=1

        else:
            reduce_eta['index'] = 0

        if reduce_eta['index'] == reduce_eta['patience']: 
            eta = eta*reduce_eta['factor']
            reduce_eta['index'] = 0
            if reduce_eta['verbose'] == 1:
                print(f'Reduced learning rate, new eta = {eta}')
            
        return eta

    def init_history(self):

        history = {'train_loss': [],
                    'train_loss_var': [],
                    'val_loss': [],
                    'val_loss_var': []}
        
        for m in self.metrics:
            history[f'train_{m.__name__}'] = []
            history[f'train_{m.__name__}_var'] = []
            history[f'val_{m.__name__}'] = []
            history[f'val_{m.__name__}_var'] = []

        return history

    def train(self, input, target, epochs, eta, lam, alpha, n_batch, validation_split = 0.5, validation_data = None, early_stopping = None, reduce_eta = None, verbose = 1, use_opt = 0, nest=False, l1_reg=False):
        
        # Checking conflicts between parameters:
        if n_batch > input.shape[1]:
            raise ValueError("n_batch can't be bigger than training set lenght.")
        
        # Inserire check su dimensioni array, ecc.

        #Initialize parameters:
        if early_stopping is not None:
            early_stopping['index'] = 0

        if reduce_eta is not None:
            reduce_eta['index'] = 0

        # Initialize history.
        history = self.init_history()
        
        if validation_data is not None:

            val_input = validation_data[0]
            val_target = validation_data[1]

            train_input = input
            train_target = target

        else:

            dim_input = input.shape[1] # number of columns in input

            validation_div = np.floor((1-validation_split)*dim_input).astype(int) # number of elements in validation set

            if n_batch > validation_div:
                raise ValueError("n_batch can't be bigger than training set lenght.")

            index = np.arange(dim_input) # array of index
            np.random.shuffle(index)
            input = input[:,index]
            target = target[:,index]

            val_input = input[:,validation_div:] # validation input
            val_target = target[:,validation_div:] # validation target

            train_input = input[:,:validation_div] # train input
            train_target = target[:,:validation_div] # train target

        index = np.arange(train_input.shape[1]) # array of index 

        j = 0

        while j < epochs:

            np.random.shuffle(index)
            input_new = train_input[:,index]
            target_new = train_target[:,index]

            for k in range(0, input_new.shape[1], n_batch):
                end_idx = min(k + n_batch, input_new.shape[1])

                self.input_layer.layer = input_new[:, k:end_idx]
                self.output_layer.target = target_new[:, k:end_idx]

                if nest:
                    self.output_layer.nest_update(alpha)
                    self.output_layer.forward(mode='nest')
                    self.output_layer.backward_nest(lossfunc = self.d_lossfunc, last = True)
                else:
                    self.output_layer.forward()
                    self.output_layer.backward(lossfunc = self.d_lossfunc, last = True)

                self.output_layer.update_weights(eta, lam, alpha, use_opt=use_opt, nest=nest, l1_reg=l1_reg)

                self.update_history_batch(history, self.output_layer.forward(), self.output_layer.target, 'train')

                self.update_history_batch(history, self.predict(val_input), val_target, 'val')

            
            #if n_batch != input_new.shape[1]:
            #    self.output_layer.reset_velocity()
                
            self.update_history_epoch(history)

            if np.isnan(history['train_loss']).any() or np.isinf(history['train_loss']).any() or np.isnan(history['val_loss']).any() or np.isinf(history['val_loss']).any():
                print("Model couldn't fit: occurred divergence!")
                return -1

            if verbose == 1:
                self.print_epoch(j, history, eta)

            if j > 1:
                
                if early_stopping is not None:
                    check, history = self.check_early_stopping(history, early_stopping)
                    if check == 1: return history

                if reduce_eta is not None:
                    eta = self.check_reduce_eta(history, reduce_eta, eta)

            j += 1
        
        return self.clear_history(history)
    
    def retrain(self, input, target, epochs, eta, lam, alpha, n_batch, test_data = None, early_stopping = None, reduce_eta = None, verbose = 1, use_opt = 0): 

        # Checking conflicts between parameters:
        if n_batch > input.shape[1]:
            raise ValueError("n_batch can't be bigger than training set lenght.")
        
        # Inserire check su dimensioni array, ecc.

        #Initialize parameters:
        if early_stopping is not None:
            early_stopping['index'] = 0

        if reduce_eta is not None:
            reduce_eta['index'] = 0

        # Initialize history.
        history = {'train_loss': [],
                    'train_loss_var': [],
                    'test_loss': [],
                    'test_loss_var': []}
        
        for m in self.metrics:
            history[f'train_{m.__name__}'] = []
            history[f'train_{m.__name__}_var'] = []
            history[f'test_{m.__name__}'] = []
            history[f'test_{m.__name__}_var'] = []

        test_input = test_data[0]
        test_target = test_data[1]

        train_input = input
        train_target = target

        index = np.arange(train_input.shape[1]) # array of index 

        j = 0

        while j < epochs:

            np.random.shuffle(index)
            input_new = train_input[:,index]
            target_new = train_target[:,index]

            for k in range(input_new.shape[1]//n_batch):

                y_pred_val = self.predict(test_input)


                self.update_history_batch(history, y_pred_val, test_target, 'test')

                if k == input_new.shape[1]//n_batch -1:
                    self.input_layer.layer = input_new[:,k*n_batch:]
                    self.output_layer.target = target_new[:,k*n_batch:]
                else:
                    self.input_layer.layer = input_new[:,k*n_batch:(k+1)*n_batch]
                    self.output_layer.target = target_new[:,k*n_batch:(k+1)*n_batch]

                y_pred_train = self.output_layer.forward()

                self.update_history_batch(history, y_pred_train, self.output_layer.target, 'train')
                
                self.output_layer.backward(lossfunc = self.d_lossfunc, last = True)
                self.output_layer.update_weights(eta, lam, alpha, use_opt)
            
            if n_batch != input_new.shape[1]:
                self.output_layer.reset_velocity()
                
            history[f'train_loss'].append(np.mean(history[f'train_loss_var']))
            history[f'test_loss'].append(np.mean(history[f'test_loss_var']))
            history[f'test_loss_var']  = []
            history[f'train_loss_var'] = []
            for m in self.metrics:
                history[f'train_{m.__name__}'].append(np.mean(history[f'train_{m.__name__}_var']))
                history[f'test_{m.__name__}'].append(np.mean(history[f'test_{m.__name__}_var']))
                history[f'train_{m.__name__}_var'] = []
                history[f'test_{m.__name__}_var'] = []

            if np.isnan(history['train_loss']).any() or np.isinf(history['train_loss']).any() or np.isnan(history['test_loss']).any() or np.isinf(history['test_loss']).any():
                print("Model couldn't fit: occurred divergence!")
                return -1

            if verbose == 1:
                self.print_epoch(j, history, eta, val_or_test='test')

            if j > 1:
                
                if early_stopping is not None:
                    check, history = self.check_early_stopping(history, early_stopping)
                    if check == 1: return history

                if reduce_eta is not None:
                    eta = self.check_reduce_eta(history, reduce_eta, eta)

            j += 1
        
        del history['train_loss_var']
        del history['test_loss_var']
        for m in self.metrics:
            del history[f'train_{m.__name__}_var']
            del history[f'test_{m.__name__}_var']
        return history
        