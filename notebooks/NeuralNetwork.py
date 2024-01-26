import numpy as np
from functions import lossfunc
import sys

class NeuralNetwork:
    def __init__(self, input_layer, output_layer, loss, metrics = []):
        """ Initialize NeuralNetwork object.

        Args:
            input_layer (Layer): input layer of the neural network.
            output_layer (Layer): output layer of the neural network.
            loss (string): loss function used for training.
            metrics (list, optional): list of additional metrics to monitor during training. Defaults to [].
        """        
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.lossfunc = lossfunc[loss][0]  # Set the loss function for training
        self.d_lossfunc = lossfunc[loss][1]  # Set the derivative of the loss function
        self.metrics = metrics

    def predict(self,input):
        """ Make prediction using the trained network.

        Args:
            input (np.array): input data.

        Returns:
            np.array: predictions made by the network.
        """        
        self.input_layer.layer = input
        return self.output_layer.forward(mode = 'predict')
    
    def update_history_batch(self, history, y_pred, y_true, train_or_val):
        """ Update training or validation history with batch-level information.

        Args:
            history (dict): dictionary storing training/validation/test history.
            y_pred (np.array): predictions made by the network.
            y_true (np.array): true values.
            train_or_val (_type_): indicates whether the update is for training, validation or test data.

        Returns:
            dict: updated history dictionary.
        """        

        history[f'{train_or_val}_loss_var'].append(self.lossfunc(y_pred, y_true))
        for m in self.metrics:
            history[f'{train_or_val}_{m.__name__}_var'].append(m(y_pred, y_true))

        return history

    def update_history_epoch(self, history, val_or_test = 'val'):
        """ Update training and validation history with epoch-level information.

        Args:
            history (dict): dictionary storing training/validation/test history.
            val_or_test (str): indicates whether the update is for training, validation or test data.

        Returns:
            dict: updated history dictionary.
        """ 
        # Update mean loss for the epoch.       
        history[f'train_loss'].append(np.mean(history[f'train_loss_var']))
        history[f'{val_or_test}_loss'].append(np.mean(history[f'{val_or_test}_loss_var']))

        # Reset batch-level loss storage.
        history[f'train_loss_var'] = []
        history[f'{val_or_test}_loss_var'] = []

        # Update mean metrics for the epoch.
        for m in self.metrics:
            history[f'train_{m.__name__}'].append(np.mean(history[f'train_{m.__name__}_var']))
            history[f'{val_or_test}_{m.__name__}'].append(np.mean(history[f'{val_or_test}_{m.__name__}_var']))

            # Reset batch-level metric storage.
            history[f'train_{m.__name__}_var'] = []
            history[f'{val_or_test}_{m.__name__}_var'] = []
        
        return history

    def print_epoch(self, j, history, eta, val_or_test='val'):
        """ Print information about the current epoch.

        Args:
            j (int): current epoch number.
            history (dict): dictionary storing training/validation history.
            eta (float): learning rate.
            val_or_test (str, optional): indicates whether printing for validation or test data. Defaults to 'val'.
        """        
        
        print(f'Epoch {j}: ', end='')
        # Print current epoch loss information.
        print(f'train_loss: {history["train_loss"][-1]:.3e}, ', end = '')
        print(f'{val_or_test}_loss: {history[f"{val_or_test}_loss"][-1]:.3e}; ', end = '')

        # Print current epoch metrics information.
        for m in self.metrics:
            print(f'train_{m.__name__}: {history[f"train_{m.__name__}"][-1]:.3e}, ', end = '')
            print(f'{val_or_test}_{m.__name__}: {history[f"{val_or_test}_{m.__name__}"][-1]:.3e}  ', end = '')

        # Print current epoch learning rate.
        print(f'lr : {eta}')

    def clear_history(self, history, val_or_test = 'val'):
        """ Remove batch-level information from history.    

        Args:
            history (dict): dictionary storing training/validation/test information.
            val_or_test (str): indicates whether the update is for validation or test data.

        Returns:
            dict: updated history dictionary without batch-level information.
        """        
        del history['train_loss_var']
        del history[f'{val_or_test}_loss_var']
        for m in self.metrics:
            del history[f'train_{m.__name__}_var']
            del history[f'{val_or_test}_{m.__name__}_var']
        return history

    def check_early_stopping(self, history, early_stopping):
        """ Check if early stopping criteria are met.

        Args:
            history (dict): dictionary storing training/validation/test information.
            early_stopping (dict): dictionary containing early stopping parameters.

        Returns:
            tuple(int, dict): (check, history), check is 1 if early stopping criteria are met, 0 otherwise.
        """        

        # Compare the last 2 history elements of early stopping monitor using early stopping compare function.
        if early_stopping['compare_function'](history[early_stopping['monitor']][-1], 
                                              history[early_stopping['monitor']][-2]):
            early_stopping['index'] +=1 
        
        # If criteria are not met reset early stopping index.
        else:
            early_stopping['index'] = 0

        # If early stopping index is equal to early stopping patience trigger early stopping.
        if early_stopping['index'] == early_stopping['patience']: 
            if early_stopping['verbose'] == 1:
                print('Early stopped')
            history = self.clear_history(history)
            return 1,history
        
        return 0,history
    
    def check_reduce_eta(self, history, reduce_eta, eta):
        """ Check if it is time to reduce the learning rate.

        Args:
            history (dict): dictionary storing training/validation/test information.
            reduce_eta (_type_): dictionary containing learning rate reduction parameters.
            eta (_type_): current learning rate.

        Returns:
            float: updated learning rate.
        """        
        # Compare the last 2 history elements of eta reduction monitor using eta reduction compare function.
        if reduce_eta['compare_function'](history[reduce_eta['monitor']][-1], history[reduce_eta['monitor']][-2]):
            reduce_eta['index'] +=1

        # If criteria are not met reset eta reduction index.
        else:
            reduce_eta['index'] = 0

        # If eta reduction index is equal to eta reduction patience update learning rate.
        if reduce_eta['index'] == reduce_eta['patience']: 
            eta = eta*reduce_eta['factor']
            reduce_eta['index'] = 0
            if reduce_eta['verbose'] == 1:
                print(f'Reduced learning rate, new eta = {eta}')
            
        return eta

    def init_history(self, val_or_test = 'val'):
        """ Initialize a dictionary for storing training/validation history.

        Returns:
            dict: initialized history dictionary.
        """        

        history = {'train_loss': [],
                    'train_loss_var': [],
                    f'{val_or_test}_loss': [],
                    f'{val_or_test}_loss_var': []}
        
        for m in self.metrics:
            history[f'train_{m.__name__}'] = []
            history[f'train_{m.__name__}_var'] = []
            history[f'{val_or_test}_{m.__name__}'] = []
            history[f'{val_or_test}_{m.__name__}_var'] = []

        return history

    def train(self, input, target, epochs, eta, lam, alpha, n_batch, validation_split = 0.5, validation_data = None, early_stopping = None, reduce_eta = None, verbose = 1, use_opt = 0, nest=False):
        """ Train the neural network.

        Args:
            input (np.array): training input data.
            target (np.array): training target data.
            epochs (int): number of training epochs.
            eta (float): learning rate.
            lam (float): regularization parameter.
            alpha (float): momentum parameter.
            n_batch (int): batch size.
            validation_split (float, optional): fraction of training data used for validation. Defaults to 0.5.
            validation_data (list(np.array, np.array), optional): list (val_input, val_target) for validation data. Defaults to None.
            early_stopping (dict, optional): dictionary containing early stopping configuration. Defaults to None.
            reduce_eta (dict, optional): dictionary containing learning rate reduction configuration. Defaults to None.
            verbose (int, optional): if 1 print information, if 0 remain silent. Defaults to 1.
            use_opt (int, optional): option to use optimization techniques. Defaults to 0.
            nest (bool, optional): option for Nestorov momentum (True: use Nestorov moemntum, False: use regular momentum). Defaults to False.

        Returns:
            dict: training history.
        """        
        # Checking conflicts between parameters:
        if n_batch > input.shape[1]:
            raise ValueError("n_batch can't be bigger than training set lenght.")

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

            dim_input = input.shape[1] # number of columns in input.

            validation_div = np.floor((1-validation_split)*dim_input).astype(int) # number of elements in validation set

            if n_batch > validation_div:
                raise ValueError("n_batch can't be bigger than training set lenght.")

            index = np.arange(dim_input) # array of index.

            # Randomize input and target arrays before slicing them in training and validation set.
            np.random.shuffle(index)
            input = input[:,index]
            target = target[:,index]

            val_input = input[:,validation_div:] # validation input.
            val_target = target[:,validation_div:] # validation target.

            train_input = input[:,:validation_div] # train input.
            train_target = target[:,:validation_div] # train target.

        index = np.arange(train_input.shape[1]) # array of index.

        j = 0

        while j < epochs:

            # Shuffle index at the beginning of every epoch.
            np.random.shuffle(index)
            input_new = train_input[:,index]
            target_new = train_target[:,index]

            for k in range(0, input_new.shape[1], n_batch):
                end_idx = min(k + n_batch, input_new.shape[1])

                # Initialize layer of input_layer.
                self.input_layer.layer = input_new[:, k:end_idx]
                # Initialize target of output_layer.
                self.output_layer.target = target_new[:, k:end_idx]

                # If nest is True update weights following Nestorov theory.
                if nest: self.output_layer.nest_update(alpha)
    
                # Perform forward and backward propagation.
                self.output_layer.forward()
                self.output_layer.backward(lossfunc = self.d_lossfunc, last = True)

                # Permorf weights update.
                self.output_layer.update_weights(eta, lam, alpha, use_opt, nest)

                # Update history dictionary with batch-level training loss and metrics information.
                history = self.update_history_batch(history, self.output_layer.forward(), self.output_layer.target, 'train')
                # Update history dictionary with batch-level validation loss and metrics information.
                history = self.update_history_batch(history, self.predict(val_input), val_target, 'val')

            # Reset velocity if batch size isn't equal to training input lenght.
            if n_batch != input_new.shape[1]:
                self.output_layer.reset_velocity()
                
            # Update history dictionary with epoch-level information.
            history = self.update_history_epoch(history)

            # Stop the training if any loss is infinite or nan.
            if np.isnan(history['train_loss']).any() or np.isinf(history['train_loss']).any() or np.isnan(history['val_loss']).any() or np.isinf(history['val_loss']).any():
                print("Model couldn't fit: occurred divergence!")
                return -1

            # Print current epoch information if verbose is 1.
            if verbose == 1:
                self.print_epoch(j, history, eta)

            # Check if early stopping or learning rate reduction criteria are met, only if the current-epoch number is higher than 1.
            if j > 1:
                
                if early_stopping is not None:
                    check, history = self.check_early_stopping(history, early_stopping)
                    if check == 1: return history

                if reduce_eta is not None:
                    eta = self.check_reduce_eta(history, reduce_eta, eta)

            j += 1
        
        # Return history dictionary cleared of batch-level information.
        return self.clear_history(history)
    
    def retrain(self, input, target, epochs, eta, lam, alpha, n_batch, test_data = None, early_stopping = None, reduce_eta = None, verbose = 1, use_opt = 0, nest=False): 
        """ Rerain the neural network.

        Args:
            input (np.array): training input data.
            target (np.array): training target data.
            epochs (int): number of training epochs.
            eta (float): learning rate.
            lam (float): regularization parameter.
            alpha (float): momentum parameter.
            n_batch (int): batch size.
            test_data (list(np.array, np.array), optional): list (test_input, test_target) for test data. Defaults to None.
            early_stopping (dict, optional): dictionary containing early stopping configuration. Defaults to None.
            reduce_eta (dict, optional): dictionary containing learning rate reduction configuration. Defaults to None.
            verbose (int, optional): if 1 print information, if 0 remain silent. Defaults to 1.
            use_opt (int, optional): option to use optimization techniques. Defaults to 0.
            nest (bool, optional): option for Nestorov momentum (True: use Nestorov moemntum, False: use regular momentum). Defaults to False.

        Returns:
            dict: training history.
        """   
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
        history = self.init_history(val_or_test='test')

        test_input = test_data[0]
        test_target = test_data[1]

        train_input = input
        train_target = target

        index = np.arange(train_input.shape[1]) # array of index 

        j = 0

        while j < epochs:

            # Shuffle index at the beginning of every epoch.
            np.random.shuffle(index)
            input_new = train_input[:,index]
            target_new = train_target[:,index]

            for k in range(0, input_new.shape[1], n_batch):
                end_idx = min(k + n_batch, input_new.shape[1])

                # Initialize layer of input_layer.
                self.input_layer.layer = input_new[:, k:end_idx]
                # Initialize target of output_layer.
                self.output_layer.target = target_new[:, k:end_idx]

                # If nest is True update weights following Nestorov theory.
                if nest: self.output_layer.nest_update(alpha)
    
                # Perform forward and backward propagation.
                self.output_layer.forward()
                self.output_layer.backward(lossfunc = self.d_lossfunc, last = True)

                # Permorf weights update.
                self.output_layer.update_weights(eta, lam, alpha, use_opt, nest)

                # Update history dictionary with batch-level training loss and metrics information.
                history = self.update_history_batch(history, self.output_layer.forward(), self.output_layer.target, 'train')
                # Update history dictionary with batch-level validation loss and metrics information.
                history = self.update_history_batch(history, self.predict(test_input), test_target, 'test')     
            
            # Reset velocity if batch size isn't equal to training input lenght.
            if n_batch != input_new.shape[1]:
                self.output_layer.reset_velocity()
                
            # Update history dictionary with epoch-level information.
            history = self.update_history_epoch(history, val_or_test='test')

            # Stop the training if any loss is infinite or nan.
            if np.isnan(history['train_loss']).any() or np.isinf(history['train_loss']).any() or np.isnan(history['test_loss']).any() or np.isinf(history['test_loss']).any():
                print("Model couldn't fit: occurred divergence!")
                return -1

            # Print current epoch information if verbose is 1.
            if verbose == 1:
                self.print_epoch(j, history, eta, val_or_test='test')

            # Check if early stopping or learning rate reduction criteria are met, only if the current-epoch number is higher than 1.
            if j > 1:
                
                if early_stopping is not None:
                    check, history = self.check_early_stopping(history, early_stopping)
                    if check == 1: return history

                if reduce_eta is not None:
                    eta = self.check_reduce_eta(history, reduce_eta, eta)

            j += 1
        
        # Return history dictionary cleared of batch-level information.
        return self.clear_history(history, val_or_test='test')
        