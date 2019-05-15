# +-----------------------------------------------------------------------+
# | convnet.py                                                            |
# | This module implements a simple ConvNet with a 2-channel input        |
# +-----------------------------------------------------------------------+


# ----------------------------
# import the necessary modules
# ----------------------------

import torch
from torch import Tensor 
from torch import nn
from torch import optim
from torch.nn import functional as F
import dlc_practical_prologue as prologue


# ----------------
# Helper functions
# ----------------

# Train model with training and test path returned
def train_model_path(model, criterion, optimizer, nb_epochs, minibatch_size, train_X, train_Y, test_X, test_Y, verbose=False):
    '''
    Train the model and return the error path
    Params:
    model 	        : defined network
    criterion       : loss function (e.g. MSE)
    optimizer       : type of optimization function (e.g. SGD)
    np_epochs       : number of epochs to train the model
    minibatch_size  : size of each minibatch
    train_X         : train input data
    train_Y         : train target data
    test_X          : test input data
    test_Y          : test target data
    verbose         : verbosity of training routine
    Returns:
    model, train_error, test_error : the trained model and the error path of the train and test set
    '''    
    train_error = []
    test_error = []
    # iterate over the epochs
    for e in range(nb_epochs):
        # iterate over the mini-batches
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
    '''
    Compute the number of errors of a given model 
    Params:
    model    	    : trained model used for prediction
    data_input      : the input data for the model
    data_target     : the target data (ground truth)
    minibatch_size  : the size of all minibatches
    Returns:
    nb_data_errors : the number of errors for the given dataset
    '''
    nb_data_errors = 0
    for b in range(0, data_input.size(0), minibatch_size):
        out = model(data_input.narrow(0, b, minibatch_size))
        # compares the outputted values for the two channels and gives back the argmax (in pred)
        _, pred = torch.max(out.data, 1)
        # calculate the number of errors in a given mini-batch
        for k in range(minibatch_size):
            if data_target[b+k] != pred[k]:
                nb_data_errors += 1
    return nb_data_errors


# Load and clean data
def load_normal_data(n_samples):
    '''
    Load and normalize the dataset. 
    Params:
    n_samples 	: number of samples to be loaded
    Returns:
    train_X, train_Y, test_X, test_Y : the data required for train and test
    '''
    train_X, train_Y, train_Class, test_X, test_Y, test_Class = prologue.generate_pair_sets(n_samples)
    mu, std = train_X.mean(), train_X.std()
    train_X.sub_(mu).div_(std)
    test_X.sub_(mu).div_(std)
    return train_X, train_Y, test_X, test_Y


# ------------------------------	
# Model Definition (as class)
# ------------------------------

class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x


# --------------------------
# Model evaluation functions
# --------------------------

def eval_convnet(verbose=False, print_error = True):
    '''
    Evaluate simple ConvNet with a 2-channel input
    Params (optional):
    verbose			: verbosity of training routines
    print_error		: print the final training error
    Returns:
    train_path, test_path : training and testing error  
    '''
    # load the normalized data 
    train_X, train_Y, test_X, test_Y = load_normal_data(1000)

    # create the convNet model   
    cnet = convNet()
    
    # train the model
    cnet, train_path, test_path = train_model_path(cnet, nn.CrossEntropyLoss(), optim.SGD(cnet.parameters(), lr=1e-1), \
                                                    50, 100, train_X, train_Y, test_X, test_Y, verbose=verbose)
    
    if print_error:
        print('Simple ConvNet Model: Final Error: {}%'.format(test_path[-1]/10))
        
    return train_path, test_path


