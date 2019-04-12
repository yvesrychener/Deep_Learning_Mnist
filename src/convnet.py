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
import src.dlc_practical_prologue as prologue


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


