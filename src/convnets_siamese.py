# +-----------------------------------------------------------------------+
# | convnets_siamese.py                                                   |
# | This module implements the siamese convnets                           |
# +-----------------------------------------------------------------------+

# ----------------------------
# import the necessary modules
# ----------------------------
import torch
from torch import Tensor 
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import dlc_practical_prologue as prologue

# ----------------
# Helper functions
# ----------------

# Compute number of Errors (for models without auxilary loss)
def compute_nb_errors(model, data_input, data_target, minibatch_size):
    nb_data_errors = 0
    for b in range(0, data_input.size(0), minibatch_size):
        out = model(data_input.narrow(0, b, minibatch_size))
        _, pred = torch.max(out.data, 1)
        for k in range(minibatch_size):
            if data_target[b+k] != pred[k]:
                nb_data_errors += 1
    return nb_data_errors

# Compute number of Errors (for models with auxilary loss)
def compute_nb_errors_aux(model, data_input, data_target, minibatch_size):
    nb_data_errors = 0
    for b in range(0, data_input.size(0), minibatch_size):
        out = model(data_input.narrow(0, b, minibatch_size))
        _, pred = torch.max(out[0].data, 1)
        for k in range(minibatch_size):
            if data_target[b+k] != pred[k]:
                nb_data_errors += 1
    return nb_data_errors

# Train model with training path returned (for models without auxilary loss)
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

# Train model with training path returned (for models with auxilary loss)
def train_model_path_aux(model, criterion, optimizer, nb_epochs, minibatch_size, train_X, train_Y, train_class, test_X, test_Y, use_al = True, verbose=False):
    train_error = []
    test_error = []
    if not use_al:
        beta = 0
    for e in range(nb_epochs):
        if use_al:
            beta = (nb_epochs - e)/nb_epochs
        for b in range(0, train_X.size(0), minibatch_size):
            out = model(train_X.narrow(0, b, minibatch_size))
            loss = criterion(out[0], train_Y.narrow(0, b, minibatch_size))
            aux_loss = criterion(out[1], train_class[:,0].narrow(0, b, minibatch_size)) + \
                            criterion(out[2], train_class[:,1].narrow(0, b, minibatch_size))
            model.zero_grad()
            ((1 - beta)*loss + beta*aux_loss).backward()
            optimizer.step()
        train_error.append(compute_nb_errors_aux(model, train_X, train_Y, minibatch_size))
        test_error.append(compute_nb_errors_aux(model, test_X, test_Y, minibatch_size))
        if(verbose): print(compute_nb_errors_aux(model, train_X, train_Y, minibatch_size))
    return model, train_error, test_error

# Load and clean data
def load_normal_data(n_samples):
    '''
    Load and normalize the dataset. 
    Params:
    n_samples 	: number of samples to be loaded
    Returns:
    train_X, train_Y, train_Class, test_X, test_Y, test_Class :
                  the data
    '''
    train_X, train_Y, train_Class, test_X, test_Y, test_Class = prologue.generate_pair_sets(n_samples)
    mu, std = train_X.mean(), train_X.std()
    train_X.sub_(mu).div_(std)
    test_X.sub_(mu).div_(std)
    return train_X, train_Y, train_Class, test_X, test_Y, test_Class

# ------------------------------	
# Model Definitions (as classes)
# ------------------------------

# 14px digit classification
class Net_14px(nn.Module):
    def __init__(self):
        super(Net_14px, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)) #image size 12x12-> image size 6x6
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)) #image size 4x4 -> image size 2x2
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

# Convnet with weight sharing (siamese network)
class siamese_Net(nn.Module):
    def __init__(self):
        super(siamese_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(200, 10)
        self.fc4 = nn.Linear(10,2)

    def forward(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1(x[:,0,:,:].view(x.size(0),1,14,14)), kernel_size=2, stride=2)) #image size 12x12-> image size 6x6
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2)) #image size 4x4 -> image size 2x2
        x1 = F.relu(self.fc1(x1.view(-1, 256)))
        x2 = F.relu(F.max_pool2d(self.conv1(x[:,1,:,:].view(x.size(0),1,14,14)), kernel_size=2, stride=2)) #image size 12x12-> image size 6x6
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2)) #image size 4x4 -> image size 2x2
        x2 = F.relu(self.fc1(x2.view(-1, 256)))
        x = F.relu(self.fc3(torch.cat((x1.view(-1,100), x2.view(-1, 100)), 1)))
        x = self.fc4(x)
        
        n1 = self.fc2(x1)
        n2 = self.fc2(x2)
        return [x, n1, n2]

# Convnet without weight sharing (still using siamese type architecture)
class siamese_Net_no_WS(nn.Module):
    def __init__(self):
        super(siamese_Net_no_WS, self).__init__()
        self.conv1a = nn.Conv2d(1, 32, kernel_size=3) 
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1a = nn.Linear(256, 100)
        self.fc2a = nn.Linear(100, 10)
        self.conv1b = nn.Conv2d(1, 32, kernel_size=3) 
        self.conv2b = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1b = nn.Linear(256, 100)
        self.fc2b = nn.Linear(100, 10)
        self.fc3 = nn.Linear(200, 10)
        self.fc4 = nn.Linear(10,2)

    def forward(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1a(x[:,0,:,:].view(x.size(0),1,14,14)), kernel_size=2, stride=2)) #image size 12x12-> image size 6x6
        x1 = F.relu(F.max_pool2d(self.conv2a(x1), kernel_size=2, stride=2)) #image size 4x4 -> image size 2x2
        x1 = F.relu(self.fc1a(x1.view(-1, 256)))
        x2 = F.relu(F.max_pool2d(self.conv1b(x[:,1,:,:].view(x.size(0),1,14,14)), kernel_size=2, stride=2)) #image size 12x12-> image size 6x6
        x2 = F.relu(F.max_pool2d(self.conv2b(x2), kernel_size=2, stride=2)) #image size 4x4 -> image size 2x2
        x2 = F.relu(self.fc1b(x2.view(-1, 256)))
        x = F.relu(self.fc3(torch.cat((x1.view(-1,100), x2.view(-1, 100)), 1)))
        x = self.fc4(x)
        
        n1 = self.fc2a(x1)
        n2 = self.fc2b(x2)
        return [x, n1, n2]

# --------------------------
# Model evaluation functions
# --------------------------
def eval_transfer_learning(verbose=False, print_error = True, freeze_feat_ext = True):
    '''
    Evaluate the 2 step transfer learning method
    Params (optional):
    verbose			: verbosity of training routines
    print_error		: print the final training error
    freeze_feat_ext	: freeze the feature extractor during second stage
    Returns:
    train_path, test_path : training and testing error 
    
    '''
    # load the data and extract the labels for the first stage of training
    train_X, train_Y, train_Class, test_X, test_Y, test_Class = load_normal_data(1000)
    
    train_target_14px = torch.cat((train_Class[:,0], train_Class[:,1]))
    train_input_14px = torch.cat((train_X[:,0,:,:].view(1000,1,14,14), train_X[:,1,:,:].view(1000,1,14,14)))
    test_target_14px = torch.cat((test_Class[:,0], test_Class[:,1]))
    test_input_14px = torch.cat((test_X[:,0,:,:].view(1000,1,14,14), test_X[:,1,:,:].view(1000,1,14,14)))
    
    # train the feature extraction model
    model_14px = Net_14px()
    model_14px, train_path, test_path = train_model_path(model_14px, nn.CrossEntropyLoss(), optim.SGD(model_14px.parameters(), lr=1e-1), 50, 100, \
                         train_input_14px, train_target_14px, test_input_14px, test_target_14px, verbose=verbose)
 
    # create the siamese model and load the parameters from the feature extractor
    sNet = siamese_Net()
    sNet.load_state_dict(model_14px.state_dict(), strict=False)

    # freeze the feature extrctor if desired
    if freeze_feat_ext:
        for p in sNet.conv1.parameters():
            p.requires_grad = False
        for p in sNet.conv2.parameters():
            p.requires_grad = False
           
    # train the final model
    sNet, train_path, test_path = train_model_path_aux(sNet, nn.CrossEntropyLoss(), optim.SGD(sNet.parameters(), lr=1e-1), \
                                                 50, 100, train_X, train_Y, train_Class, test_X, test_Y, use_al = False, verbose=verbose)
    if print_error:
        print('Transfer Learning Model: Final Error: {}%'.format(test_path[-1]/10))
        
    return train_path, test_path

def eval_WS_no_AL(verbose=False, print_error = True):
    '''
    Evaluate the weight sharing model, no augmented loss
    Params (optional):
    verbose			: verbosity of training routines
    print_error		: print the final training error
    Returns:
    train_path, test_path : training and testing error 
    
    '''
    # load the data and extract the labels for the first stage of training
    train_X, train_Y, train_Class, test_X, test_Y, test_Class = load_normal_data(1000)
    
    # create the siamese model and train it
    sNet = siamese_Net()
    sNet, train_path, test_path = train_model_path_aux(sNet, nn.CrossEntropyLoss(), optim.SGD(sNet.parameters(), lr=1e-1), \
                                                 50, 100, train_X, train_Y, train_Class, test_X, test_Y, use_al = False, verbose=verbose)
    
    if print_error:
        print('Weight-Sharing, no Augmented Loss: Final Error: {}%'.format(test_path[-1]/10))
        
    return train_path, test_path
    
def eval_WS_AL(verbose=False, print_error = True):
    '''
    Evaluate the weight sharing model with augmented loss
    Params (optional):
    verbose			: verbosity of training routines
    print_error		: print the final training error
    Returns:
    train_path, test_path : training and testing error 
    
    '''
    # load the data and extract the labels for the first stage of training
    train_X, train_Y, train_Class, test_X, test_Y, test_Class = load_normal_data(1000)
    
    # create the siamese model and train it
    sNet = siamese_Net()
    sNet, train_path, test_path = train_model_path_aux(sNet, nn.CrossEntropyLoss(), optim.SGD(sNet.parameters(), lr=1e-1), \
                                                 50, 100, train_X, train_Y, train_Class, test_X, test_Y, use_al = True, verbose=verbose)
    if print_error:
        print('Weight-Sharing, with Augmented Loss: Final Error: {}%'.format(test_path[-1]/10))
        
    return train_path, test_path

def eval_no_WS_no_AL(verbose=False, print_error = True):
    '''
    Evaluate the model without weight sharing and without augmented loss
    Params (optional):
    verbose			: verbosity of training routines
    print_error		: print the final training error
    Returns:
    train_path, test_path : training and testing error 
    
    '''
    # load the data and extract the labels for the first stage of training
    train_X, train_Y, train_Class, test_X, test_Y, test_Class = load_normal_data(1000)
    
    # create the siamese model and train it
    sNet = siamese_Net_no_WS()
    sNet, train_path, test_path = train_model_path_aux(sNet, nn.CrossEntropyLoss(), optim.SGD(sNet.parameters(), lr=1e-1), \
                                                 50, 100, train_X, train_Y, train_Class, test_X, test_Y, use_al = False, verbose=verbose)
    if print_error:
        print('No Weight-Sharing, no Augmented Loss: Final Error: {}%'.format(test_path[-1]/10))
        
    return train_path, test_path

def eval_no_WS_AL(verbose=False, print_error = True):
    '''
    Evaluate the model without weight sharing and without augmented loss
    Params (optional):
    verbose			: verbosity of training routines
    print_error		: print the final training error
    Returns:
    train_path, test_path : training and testing error 
    
    '''
    # load the data and extract the labels for the first stage of training
    train_X, train_Y, train_Class, test_X, test_Y, test_Class = load_normal_data(1000)
    
    # create the siamese model and train it
    sNet = siamese_Net_no_WS()
    sNet, train_path, test_path = train_model_path_aux(sNet, nn.CrossEntropyLoss(), optim.SGD(sNet.parameters(), lr=1e-1), \
                                                 50, 100, train_X, train_Y, train_Class, test_X, test_Y, use_al = True, verbose=verbose)
    if print_error:
        print('No Weight-Sharing, with Augmented Loss: Final Error: {}%'.format(test_path[-1]/10))
        
    return train_path, test_path































