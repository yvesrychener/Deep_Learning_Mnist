# +-----------------------------------------------------------------------+
# | plotting.py                                                           |
# | This module implements plotting the results of the networks           |
# +-----------------------------------------------------------------------+

import matplotlib.pyplot as plt
import numpy as np

def plot_result(eval_f, n_repe, title):
    '''
    Plot the network training path and testing path
    Params:
    eval_f  : evaluation function of the network
    n_repe  : number of repetitions
    title   : title for the plot
    '''
    test_e = []
    train_e = []
    
    # evaluate the models
    for i in range(n_repe):
        train, test = eval_f()
        test_e.append(test)
        train_e.append(train)
    
    # calculate the statistics
    train_mean = np.array(train_e).mean(0)
    test_mean = np.array(test_e).mean(0)
    train_std = np.array(train_e).std(0)
    test_std = np.array(test_e).std(0)
    
    # print final average train and test errors
    print('Final average train error: {}%'.format(train_mean[-1]/10))
    print('Final average test error: {}%'.format(test_mean[-1]/10))

    # plotting
    plt.subplot(1,2,1)
    plt.plot(train_mean)
    ub = train_mean + 1.96 * train_std
    lb = train_mean - 1.96 * train_std
    plt.fill_between(range(train_mean.shape[0]), ub, lb, alpha=0.5)
    plt.title('Training Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (out of 1000)')
    
    plt.subplot(1,2,2)
    plt.plot(test_mean)
    ub = test_mean + 1.96 * test_std
    lb = test_mean - 1.96 * test_std
    plt.fill_between(range(test_mean.shape[0]), ub, lb, alpha=0.5)
    plt.title('Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (out of 1000)')
    
    plt.suptitle(title)
    return train_mean, test_mean, train_std, test_std
   