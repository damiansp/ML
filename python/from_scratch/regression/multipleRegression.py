import matplotlib.pyplot as plt
import numpy as np
import sframe

# Load in data
sales = sframe.SFrame('../../../data/kc_house_data.csv')



# Create additional variables
sales['bedrooms_squared'] = sales['bedrooms'] ** 2
sales['bed_bath'] = sales['bedrooms'] * sales['bathrooms']
sales['log_sqft_living'] = np.log(sales['sqft_living'])
# A deliberately nonsensical variable:
sales['lat_plus_long'] = sales['lat'] + sales['long']

#print sales[0]

# Divide into testing and training data
train, test = sales.random_split(0.8, seed = 0)



def get_rss(predictions, Y):
    '''
    @param predictions (1-D array):
       the predicted values of the model being tested
    @param Y (1-D array):
       the true (target) values of the response variable being predicted
    @return rss (float):
       the residual sum of squares (rss)
    '''
    error = predictions - Y
    rss = sum(error ** 2)
    return rss


def get_numpy_data(data_sframe, features, Y, verbose = False):
    '''
    Convert an SFrame into a numpy matrix and response variable (Y) to a 
    numpy array.
    @param data_sframe (SFrame):
       the input data in SFrame format
    @param features (1-D array of stings):
       names of features to be included
    @param Y (string):
       name of column to be treated as response variable
    @param verbose (bool):
       if True prints first few rows of features (as sframe)
    @return (model_matrix, Y_array)
       model matrix as an numpy matrix;
       Y (response variable) as a numpy array
    '''
    data_sframe['intercept'] = 1 # bias or intercept term
    features = ['intercept'] + features
    features_sframe = data_sframe[features]

    if verbose:
        print features_sframe.head()

    # convert features_sframe to numpy matrix
    model_matrix = features_sframe.to_numpy()

    # convert response to np array
    Y_array = data_sframe[Y].to_numpy()

    return (model_matrix, Y_array)


(example_features, example_Y) = get_numpy_data(sales, ['sqft_living'], 'price')
#print example_features[:3, :]
#print example_Y[:3]



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

# Assign test weights
my_weights = np.array([1., 1.])
test_predictions = predict(example_features, my_weights)
#print test_predictions




def feature_derivative(error, feature):
    '''
    Compute the derivative of the cost function (rss)
    The cost function, RSS, is:
    sum(error)^2 or, for a single feature, it is:
    (error)^2 = (w[0]X[, 0] + w[1]X[, 1] + ... + w[d]X[, d])^2 (for d features)
    the derivative (wrt the ith weight) of which is:
    2(w[0]X[,0] + w[1]X[, 1] + ... + w[d]X[, d])(X[, i]
    = 2(error)(X[, i])
    
    @param error (1-D numpy array):
       model errors for a given set of weight (coefficients)
    @param feature (1-D numpy array):
       must be of the same lenght as error; a column (single feature) of the
       model matrix (X)
    @return derivative (float)
    '''
    derivative = 2 * np.dot(error, feature)
    return derivative

# Test
#my_weights = np.array([0., 0.])
#test_predictions = predict(example_features, my_weights)
#errors = test_predictions - example_Y
#feature = example_features[:, 0]
#derivative = feature_derivative(errors, feature)
#print derivative
#print -np.sum(example_Y) * 2 # should be the same as derivative


# The Gradient Descent Algorithm
def regression_gradient_descent(X, Y, W_init, eta, tolerance, verbose = False):
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

    while not converged:
        # prediction from current weights
        preds = predict(X, W)
        error = preds - Y

        # initialize gradient sum of squares
        gradient_ss = 0

        # update each features weight
        for i in range(len(W)):
            derivative = feature_derivative(error, X[:, i])

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

    return W



# Test
simple_features = ['sqft_living']
my_Y = 'price'
(simple_feature_matrix, Y) = get_numpy_data(train, simple_features, my_Y)
initial_w = np.array([-47000., 1.])
eta = 7e-12
tolerance = 2.5e7

mod1_w = regression_gradient_descent(
    simple_feature_matrix, Y, initial_w, eta, tolerance)
print 'final weights:', mod1_w


# Use optimized weights to predict values in the test set
(simple_features_matrix_test, Y_test) = get_numpy_data(
    test, simple_features, my_Y)

mod1_preds = predict(simple_features_matrix_test, mod1_w)


# Visualize the fit
plt.plot(test['sqft_living'], test['price'], 'o',
         test['sqft_living'], mod1_preds, '-')
plt.show()

# And get RSS
mod1_rss = get_rss(mod1_preds, test['price'])
print 'mod1 rss:', mod1_rss # 2.75400044902e+14



# Now run a more complex model and compare the goodness of fit (RSS)
complex_features = ['bedrooms',
                    'bathrooms',
                    'sqft_living',
                    'floors',
                    'grade',
                    'yr_built',
                    'sqft_living15']
(complex_train, Y_train) = get_numpy_data(train, complex_features, my_Y)
(complex_test, Y_test)   = get_numpy_data(test,  complex_features, my_Y)

eta = 4e-12
tolerance = 1e11
initial_w = np.array([-1000., 1., 1., 1., 1., 1., 1., 1.])
mod2_w = regression_gradient_descent(
    complex_train, Y_train, initial_w, eta, tolerance)

mod2_preds = predict(complex_test, mod2_w)
mod2_rss = get_rss(mod2_preds, test['price'])
print 'mod2 rss:', mod2_rss # 2.69161743496e+14 (improved over simpler model)









