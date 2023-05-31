from __future__ import division


def predict_probability_logistic(X, W):
    '''
    Produce probability estimates for P(y[i] = 1 | x[i], W) =
    1 / (1 + exp(-WTh(x[i]))), where WT is the transpose of W, and h(x[i]) 
    is the ith feature (predictor). Output on [0, 1].

    @param X (matrix):
      the model matrix
    @param W (vector):
      the model weights (coefficients)

    @return predictions (vector)
      the predicted values of Y (response variable) for each row in X
    '''

    dot_prod = np.dot(X, W)

    # P(y[i] = 1 | x[i], W):
    predictions = 1 / (1 + np.exp(-dot_prod))

    return predictions

def logistic_feature_derivative(errors, feature):
    '''
    Compute the derivative of the log likelihood wrt a single coefficient:
    dL/dw[j] = sum[i]((h[j](x[i])(1(y[i] = 1) - P(y[i] = 1 | x[i], W)))
    or         sum[i]((ith feature)(error)) and 1() is the indicator function

    @param errors (vector):
      the difference in the indicator function and the predicted probability
      for each y[i] (see above)
    @param feature (vector):
      the vector of all values for the feature column of X
    '''
    derivative = sum(feature * errors)
    # alt:
    #derivative = sum(np.dot(feature, errors))

    return derivative


def compute_logistic_log_likelihood(X, Y, W):
    '''
    Compute the log-likelihood of a set of weights for a logistic regression
    model where the log-likelihood:
    L(W) = sum((1(y[i] = 1) - 1)WTh(x[i]) - ln(1 + exp(-WTh(x[i]))))

    @param X (matrix):
      the model matrix
    @param Y (vector of binary values):
      the response variable (1 for positive cases, 0 otherwise)
    @param W (vector):
      the weights (coefficients)

    @return ll (float):
      the log-likelihood
    '''

    indicator = (Y == 1)
    scores = np.dot(X, W)
    log_exp = np.log(1. + np.exp(-scores))

    # Check to prevent overflow
    mask = np.isinf(log_exp)
    log_exp[mask] = -score[mask]
    ll = np.sum((indicator - 1) * scores - log_exp)

    return ll

def logisitic_regression(X, Y, W_init, eta, max_iter, verbose = False):
    '''
    Find an optimal set of weights to maximize the (log) likelihood of the data
    in Y given X
    
    @param X (matrix):
      the model matrix
    @param Y (vector of binary values):
      the response variable (where 1 indicates a match)
    @param W_init (vector):
      initial values for weights (e.g. all 0s or an educated guess)
    @param eta (float):
      the gradient step size ("learning rate")
    @param max_iter (int):
      number of iterations to run gradient ascent
    @param verbose (bool):
      if True, prints iteration and log-likehood at iteration
    
    @return W (vector):
      the optimized weights (coefficients)
    '''
    W = np.array(W_init)

    for itr in xrange(max_iter):
        # Predict P(y[i] = 1 | x[i], W)
        predictions = predict_probability_logit(X, W)
        # Compute indicator for y[i] = 1
        indicator = (Y == 1)
        # Compute errors
        errors = indicator - predictions

        # loop over each coef
        for j in xrange(len(W)):
            derivative = logistic_feature_derivative(X[:, j], errors)

        # Update weights
        W[j] += (eta * derivative)

    if verbose:
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or \
           (itr <= 1000 and itr % 100 == 0) or \
           (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_logistic_log_likelihood(X, Y, W)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
                                            
    return W



