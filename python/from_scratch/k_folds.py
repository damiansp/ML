import numpy as np
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'regression/'))
from regression import *


def k_folds_regression(k,
                       X,
                       Y,
                       eta,
                       tolerance,
                       max_iterations,
                       l2_penalty = 0):
    '''
    Split the data in the model matrix into k folds and use the hold-out group
    in each for cross validation.

    @param k (int):
      number of folds (1 is fastest; num_rows in X is optimal but slowest)
    @param X (matrix):
      the model matrix (observations x features)
    @param Y (vector):
      the response variable
    @param features_list (list of strings):
      the features in X to be used in the model
    @params l1_penalty, l2_penalty (float on [0, inf))
      the parameter (lambda) for l1 and l2 penalties to the cost function; if
      both are 0, it is equivalent to ordinary least squares (OLS; RSS).

    @return total_rss / k (float):
      the mean rss over all folds
    '''

    n = len(X)
    fold_size = n / k
    total_rss = 0 # init

    for fold in xrange(k):
        fold_start = fold * fold_size

        # hold out a validation set
        X_validation = X[fold_start:(fold_start + fold_size)]
        Y_validation = Y[fold_start:(fold_start + fold_size)]

        # the test set is then all data except the validation set
        # data before the validation set:
        train_pre = X[0:fold_start, :]
        Y_pre = Y[0:fold_start]
        # and after it:
        train_post = X[(fold_start + fold_size):n, :]
        Y_post = Y[(fold_start + fold_size):n]
        # merge pre and post:

        X_train = np.vstack((train_pre, train_post))
        Y_train = np.hstack((Y_pre, Y_post))
        
        weights = regression_gradient_descent(X = X_train,
                                              Y = Y_train,
                                              W_init = np.zeros(len(X[0, :])),
                                              eta = eta,
                                              tolerance = tolerance,
                                              max_iterations = max_iterations,
                                              l2_penalty = l2_penalty)

        # Test model on the validation group
        predictions = predict(X_validation, weights)
        error = predictions - Y_validation
        total_rss += sum(error ** 2)

    return total_rss / k



# TO DO: Generalize k_folds to work with any model
# Requires that all models output predictions (GD and CD currently output
# optimized weights only)
def k_folds(k, model, X, Y):
    '''
    Perform k-folds cross validation on a model using model matrix X and 
    response variable Y

    @param k (int): 
      number of folds (k = n_row(X) is optimal but slowest)
    @param model (dict):
      a dictionary with model: name_of_model, and parameters to model passed in
    @param X (matrix):
      the model matrix
    @param Y (vector):
      the response variable values

    @return total_rss / k (float):
    the mean rss over all folds
    '''

    n = len(X)
    fold_size = n / k
    total_rss = 0 # init
    
    for fold in xrange(k):
        fold_start = fold * fold_size
        
        # hold out a validation set                                            
        X_validation = X[fold_start:(fold_start + fold_size)]
        Y_validation = Y[fold_start:(fold_start + fold_size)]

        # the test set is then all data except the validation set              
        # data before the validation set:                                      
        train_pre = X[0:fold_start, :]
        Y_pre = Y[0:fold_start]
        # and after it:                                                        
        train_post = X[(fold_start + fold_size):n, :]
        Y_post = Y[(fold_start + fold_size):n]
        # merge pre and post:                                                  
        X_train = np.vstack((train_pre, train_post))
        Y_train = np.hstack((Y_pre, Y_post))

        # Test model on the validation group
        predictions = '''TO DO'''
        error = predictions - Y_validation
        total_rss += sum(error ** 2)

    return total_rss / k
