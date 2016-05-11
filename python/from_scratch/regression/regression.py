import numpy as np
import sframe



# Predict
def predict(X, coeffs):
    '''
    Use a model matrix X and given coefficients to predict the output of a 
    linear model.
    @param X (numpy matrix):
       the model matrix
    @param coeffs (numpy 1-D array):
       the coefficients (weights)
    @return predictions (numpy 1-D array):
       the models predicted values
    '''
    predictions = np.dot(X, coeffs)
    return predictions


def feature_derivative(
        error, feature, weight, feature_is_constant, l2_penalty = None):
    '''
    Compute the derivative of the cost function (rss)
    The cost function, RSS, is:
    sum(error)^2 or, for a single feature, it is:
    (error)^2 = (w[0]X[, 0] + w[1]X[, 1] + ... + w[d]X[, d])^2 (for d features)
    the derivative (wrt the ith weight) of which is:
    2(w[0]X[,0] + w[1]X[, 1] + ... + w[d]X[, d])(X[, i]
    = 2(error)(X[, i])

    If there is an l2_penalty, the cost function is updated as
    RSS + lambda * ||w||[2]^2  where:
    ||w||[2]^2 = w[1]^2 + w[2]^2 + ... w[d]^2, and its derivative is simply:
    2 * lambda * w
    
    @param error (1-D numpy array):
       model errors for a given set of weight (coefficients)
    @param feature (1-D numpy array):
       must be of the same lenght as error; a column (single feature) of the
       model matrix (X)
    @param weight (float):
       for penalty derivatives only: the weight with respect to which the 
       derivative is being calculated
    @param feature_is_constant (bool):
       if true assumes the feature is the bias, and hence no l2 normalization
       occurs
    @param l2_penalty (float):
       the value of lambda for the l2 normalization
    @return derivative (float)
    '''
    derivative = 2 * np.dot(error, feature)

    if l2_penalty and not feature_is_constant:
        derivative += (2 * l2_penalty * weight)
        
    return derivative


# The Gradient Descent Algorithm
def regression_gradient_descent(X,
                                Y,
                                W_init,
                                eta,
                                tolerance,
                                max_iterations = 100,
                                l2_penalty = None,
                                verbose = False):
    '''
    Runs the gradient descent algorithm to optimize the set of weights
    (coefficients) that minimize the cost function (RSS).
    @param X (numpy matrix):
       the model matrix
    @param Y (numpy 1D array):
       the response variable
    @param W_init (numpy 1D array, length = no. columns in X):
       initial values for weigths (coefficients)
    @param eta (float):
       gradient step size, also called the "learning rate"
    @param tolerance (float):
       size of change in gradient sufficiently close to 0 to consider the 
       algorithm to have converged
    @param verbose (bool):
       if True, the weights are output at each iteration
    @return W (numpy 1D array):
       the optimized weights
    '''
    converged = False
    W = np.array(W_init)
    iterations = 1

    while not converged and iterations < max_iterations:
        # prediction from current weights
        preds = predict(X, W)
        error = preds - Y

        # initialize gradient sum of squares
        gradient_ss = 0

        # update each features weight
        for i in range(len(W)):
            if i == 0:
                feature_is_constant = True
            else:
                feature_is_constant = False

            derivative = feature_derivative(
                error, X[:, i], W[i], feature_is_constant, l2_penalty)

            # add deriv^2 to gradient_ss
            gradient_ss += (derivative ** 2)

            # subtract step size * deriv from current weight to update
            W[i] -= (eta * derivative)

        if verbose:
            print 'weights:', W

        # compute the gradient magnitude
        gradient_magnitude = np.sqrt(gradient_ss)

        if gradient_magnitude < tolerance:
            converged = True

        iterations += 1

    return W


