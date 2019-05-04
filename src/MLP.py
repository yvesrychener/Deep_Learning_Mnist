# +-----------------------------------------------------------------------+
# | MLP.py                                                                |
# | This module implements two versions of MLP (shallow & deep)           |
# +-----------------------------------------------------------------------+


# ----------------------------
# import the necessary modules
# ----------------------------

import torch
from torch import Tensor 
from torch import nn
from torch import optim
from torch.nn import functional as F
import src.dlc_practical_prologue as prologue

# layer width was based on the same rule for both shallow and deep networks:
# at each layer, the width was increased by 20% and rounded to the next integer value


# ----------------
# Helper functions
# ----------------

# Train model with training and test path returned
def train_model_path(model, criterion, optimizer, nb_epochs, minibatch_size, train_X, train_Y, test_X, test_Y, verbose=False):
    train_error = []
    test_error = []
    for e in range(nb_epochs):
        for b in range(0, train_X.size(0), minibatch_size):
            out = model(train_X.narrow(0, b, minibatch_size))
            loss = criterion(out, train_Y.narrow(0, b, minibatch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        train_error.append(compute_nb_errors(model, train_X, train_Y, minibatch_size))
        test_error.append(compute_nb_errors(model, test_X, test_Y, minibatch_size))
        if(verbose): print(compute_nb_errors(model, train_X, train_Y, minibatch_size))
    return model, train_error, test_error

# Compute number of Errors
def compute_nb_errors(model, data_input, data_target, minibatch_size):
    nb_data_errors = 0
    for b in range(0, data_input.size(0), minibatch_size):
        out = model(data_input.narrow(0, b, minibatch_size))
        # compares the outputted values for the two channels and gives back the argmax (in pred)
        _, pred = torch.max(out.data, 1)
        for k in range(minibatch_size):
            if data_target[b+k] != pred[k]:
                nb_data_errors += 1
    return nb_data_errors


# Load and clean data
def load_normal_data(n_samples):
    '''
    Load, normalize and flatten the dataset. 
    Params:
    n_samples 	: number of samples to be loaded
    Returns:
    train_X, train_Y, test_X, test_Y : the data required for train and test
    '''
    train_X, train_Y, train_Class, test_X, test_Y, test_Class = prologue.generate_pair_sets(n_samples)
    
    # normalize the data
    mu, std = train_X.mean(), train_X.std()
    train_X.sub_(mu).div_(std)
    test_X.sub_(mu).div_(std)

    # flatten the data
    train_X = train_X.view(train_X.size(0), -1)
    test_X = test_X.view(test_X.size(0), -1)

    return train_X, train_Y, test_X, test_Y


# ------------------------------	
# Model Definitions (as classes)
# ------------------------------

class MLP_shallow(nn.Module):
    """Neural Network definition, shallow Multi Layer Perceptron (1 hidden layer)"""
    def __init__(self):
        super(MLP_shallow, self).__init__()
        self.l1 = nn.Linear(392, 470)    
        self.l2 = nn.Linear(470, 2)    
 
    def forward(self, x):
        h1 = F.relu(self.l1(x))
        y = self.l2(h1)
        return y


class MLP_deep(nn.Module):
    """Neural Network definition, deep Multi Layer Perceptron (4 hidden layers)"""
    def __init__(self):
        super(MLP_deep, self).__init__()
        self.l1 = nn.Linear(392, 470)  
        self.l2 = nn.Linear(470, 564)
        self.l3 = nn.Linear(564, 677)
        self.l4 = nn.Linear(677, 812)
        self.l5 = nn.Linear(812, 2)    
 
    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        y = self.l5(h4)
        return y


# --------------------------
# Model evaluation function
# --------------------------

def eval_MLP(deep=False, verbose=False, print_error = True):
    '''
    Evaluate shallow and deep MLP models
    Params (optional):
    verbose			: verbosity of training routines
    print_error		: print the final training error
    deep        	: specify the type of model (shallow or deep)
    Returns:
    train_path, test_path : training and testing error  
    '''

    # load the data 
    train_X, train_Y, test_X, test_Y = load_normal_data(1000)

    # define model structure   
    if not deep:
        mlp = MLP_shallow()
    else:
        mlp = MLP_deep()
    
    # train the model
    mlp, train_path, test_path = train_model_path(mlp, nn.CrossEntropyLoss(), optim.SGD(mlp.parameters(), lr=1e-1), 50, 100, \
                                                    train_X, train_Y, test_X, test_Y, verbose=verbose)
    
    if print_error:
        if not deep:
            print('Shallow MLP Model: Final Error: {}%'.format(test_path[-1]/10))
        else:
            print('Deep MLP Model: Final Error: {}%'.format(test_path[-1]/10))
        
    return train_path, test_path