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


def feature_derivative(error,
                       feature,
                       weight,
                       feature_is_constant,
                       l2_penalty = None):
    '''
    Compute the derivative of the cost function (rss)
    The cost function, RSS, is:
    sum(error)^2 or, for a single feature, it is:
    (error)^2 = (w[0]X[, 0] + w[1]X[, 1] + ... + w[d]X[, d])^2 (for d features)
    the derivative (wrt the ith weight) of which is:
    2(w[0]X[,0] + w[1]X[, 1] + ... + w[d]X[, d])(X[, i]
    = 2(error)(X[, i])

    If there is an l2_penalty (AKA, ridge regression), the cost function is 
    updated as:
    RSS + lambda * ||w||[2]^2,  where ||w||[2]^2 is the l2-norm squared:
    ||w||[2]^2 = w[1]^2 + w[2]^2 + ... w[d]^2, and its derivative is simply:
    2 * lambda * w
    ...where lambda is the l2_penalty. 
    This has the effect of decreasing the coefficient size for all features as
    lambda increase.  Note that w[0] (the bias or intercept  term) is not
    reguralized.
    
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


# The Gradient Descent Algorithm for Linear Models
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


# The Coordinate Descent Algorithm for Lasso Regression on Linear Models
# First a single step:
def lasso_coordinate_descent_step(i, X, Y, W, l1_penalty):
    '''
    A single step in the coordinate descent algorithm to optimize lasso 
    regression weights (coefficients) for linear models.

    If a regression model has an l1_penalty (AKA, lasso regression), the cost
    function is updated as:
    RSS + lambda * ||w||[1], where ||w||[1] is the l1-norm:
    ||w||[1] = |w[0]| + |w[1]| + ... + |w[d]|, and its subderivative with 
    respect to the ith weight is:
      rho[i]            if i = 0 (bias or intercept: do not regularize)
      rho[i] + lambda/2 if rho[i] < -lambda/2
      rho[i] - lambda/2 if rho[i] > lambda/2
      0                 if -lambda/2 < rho[i] < lambda/2
    ...where lambda is the l1_penalty, and: 
      rho[i] = sum(feature[i] * (Y - prediction + W[i] * feature[i]))

    This has the effect of driving more coefficients to 0 as lambda increases.

    *** NOTE: Assumes that the feature is normalized. ***
    
    @param i (int):
      the index for the column in X to be updated
    @param X (matrix):
      the model matrix
    @param Y (vector):
      the response variable
    @param W (vector of length = columns in X):
      the (current) weights (coefficients)
    @param l1_penalty (float):
      the parameter (lambda) [0, inf) that controls the number of coefficients
      to be driven to 0.

    @return new_weight_i (float):
      the updated value for the ith weight
    '''
    # Compute prediction
    predictions = predict(X, W)

    # Compute rho[i]
    X_i = X[:, i]
    rho_i = sum(X_i * (Y - predictions + W[i] * X_i))

    # Update weight
    if i == 0:
        # intercept: do not regularize
        new_weight_i = rho_i
    elif rho_i < -l1_penalty / 2.:
        new_weight_i = rho_i + l1_penalty / 2.
    elif rho_i > l1_penalty/2.:
        new_weight_i = rho_i - l1_penalty / 2.
    else:
        new_weight_i = 0.

    return new_weight_i

# Now incorporate the single lasso step into an iterative function
def lasso_coordinate_descent(X, Y, W_init, l1_penalty, tolerance):
    '''                                                                        
    The full coordinate descent algorithm to optimize lasso regression weights 
    (coefficients) for linear models.                        
    
    *** NOTE: Assumes that the feature is normalized. ***                
      
    @param X (matrix):                                                         
      the model matrix                                                         
    @param Y (vector):                                                         
      the response variable                                                    
    @param W (vector of length = columns in X):                                
      the (current) weights (coefficients)                                     
    @param l1_penalty (float):                                                 
      the parameter (lambda) [0, inf) that controls the number of coefficients 
      to be driven to 0.                                        
    @param tolerance (float):
      number sufficiently close to zero for the maximum step size per iteration
      for the algorithm to be considered to have converged.
                                                                               
    @return weights (vector)
      the optimized weights (coefficients)
    '''
    # Initialize max_step: the largest step taken along the subgradient in each
    # iteration
    max_step = 1e10
    # Weights (Coeffs)
    W = W_init

    while max_step > tolerance:
        # Reset max step inside each iteration
        max_step = 0

        for i in range(len(W)):
            # Keep old W so change can be determined
            old_weight = W[i]
            W[i] = lasso_coordinate_descent_step(i, X, Y, W, l1_penalty)
            coord_change = abs(W[i] - old_weight)

            if coord_change > max_step:
                max_step = coord_chnage

    return W
