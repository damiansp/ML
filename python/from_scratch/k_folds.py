def k_folds_regression(k, X, Y, features_list, l1_penalty = 0, l2_penalty = 0):
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
        validation = X[fold_start:(fold_start + fold_size)]
        Y_validation = Y[fold_start:(fold_start + fold_size)]

        # the test set is then all data except the validation set
        # data before the validation set:
        test_pre = X[0:fold_start]
        Y_pre = Y[0:fold_start]
        # and after it:
        test_post = X[(fold_start + fold_size):n]
        Y_post = X[(fold_start + fold_size):n]
        # merge pre and post:
        test = test_pre.append(test_post)
        Y_test = Y_pre.append(Y_post)
        
        # TO DO: once gradient descent for l1 and l2 penalties has been
        # completed:
        # mod = ...

        # predictions = ...

        error = predictions - Y_test
        total_rss += sum(error ** 2)

    return total_rss / k
