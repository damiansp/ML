import matplotlib.pyplot as plt
import numpy as np
import sframe
import os, sys
import regression as reg

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                          '../'))
import utilityFunctions as uf
from k_folds import k_folds_regression


# Load some data
sales = sframe.SFrame('../../../data/kc_house_data.csv')

# Create additional variables
sales['bedrooms_squared'] = sales['bedrooms'] ** 2
sales['bed_bath'] = sales['bedrooms'] * sales['bathrooms']
sales['log_sqft_living'] = np.log(sales['sqft_living'])
# A deliberately nonsensical variable:
sales['lat_plus_long'] = sales['lat'] + sales['long']

print 'sales[0]:', sales[0]

# Divide into testing and training data
train, test = sales.random_split(0.8, seed = 0)



# A simple case with just one predictor:
(example_features, example_Y) = uf.get_numpy_data(
    sales, ['sqft_living'], 'price')
print 'example_features[:3, :]:'
print example_features[:3, :]
print 'example_Y[:3]:', example_Y[:3]


# Use predermined weight (coefficients):
my_weights = np.array([1., 1.])
test_predictions = reg.predict(example_features, my_weights)
print 'test_predictions:', test_predictions

# Test
simple_features = ['sqft_living']
(simple_feature_matrix, Y) = uf.get_numpy_data(train, simple_features, 'price')
initial_w = np.array([-47000., 1.])
eta = 7e-12
tolerance = 2.5e7

mod1_w = reg.regression_gradient_descent(
    simple_feature_matrix, Y, initial_w, eta, tolerance)
print 'final weights:', mod1_w

# Use optimized weights to predict values in the test set
(simple_features_matrix_test, Y_test) = uf.get_numpy_data(
    test, simple_features, 'price')

mod1_preds = reg.predict(simple_features_matrix_test, mod1_w)
# Visualize the fit
plt.plot(test['sqft_living'], test['price'], 'o',
         test['sqft_living'], mod1_preds, '-')
plt.show()

# And get RSS
mod1_rss = uf.get_rss(mod1_preds, test['price'])
print 'mod1 rss:', mod1_rss # 2.75400044902e+14



# Now run a more complex model and compare the goodness of fit (RSS)
complex_features = ['bedrooms',
                    'bathrooms',
                    'sqft_living',
                    'floors',
                    'grade',
                    'yr_built',
                    'sqft_living15']
(complex_train, Y_train) = uf.get_numpy_data(train, complex_features, 'price')
(complex_test, Y_test)   = uf.get_numpy_data(test,  complex_features, 'price')

eta = 4e-12
tolerance = 1e11
initial_w = np.array([-1000., 1., 1., 1., 1., 1., 1., 1.])
mod2_w = reg.regression_gradient_descent(
        complex_train, Y_train, initial_w, eta, tolerance)

mod2_preds = reg.predict(complex_test, mod2_w)
mod2_rss = uf.get_rss(mod2_preds, Y_test)
print 'mod2 rss:', mod2_rss # 2.69161743496e+14 (improved over simpler model)





# Ridge Regression: Add l2_penalty-----------------------------------------
mod3_w = reg.regression_gradient_descent(complex_train,
                                         Y_train,
                                         initial_w,
                                         eta,
                                         tolerance,
                                         max_iterations = 100,
                                         l2_penalty = 100)
mod3_preds = reg.predict(complex_test, mod3_w)
mod3_rss = uf.get_rss(mod3_preds, Y_test)
print 'mod3 rss:', mod3_rss # 2.69161743447e+14 (a slight improvement)

mod4_w = reg.regression_gradient_descent(complex_train,
                                         Y_train,
                                         initial_w,
                                         eta,
                                         tolerance,
                                         max_iterations = 100,
                                         l2_penalty = 10000)
mod4_preds = reg.predict(complex_test, mod4_w)
mod4_rss = uf.get_rss(mod4_preds, Y_test)
print 'mod4 rss:', mod4_rss # 2.69161738627ee+14 (a further slight improvement)




# Use k-folds cross validation to tune l2_penalty parameter (lambda),
# with k = 10

# set lambda as [0, 1, 10, 100, ... 1e8]
l2_lambdas = np.logspace(start = 0, stop = 8, num = 9)
l2_lambdas = np.hstack((0, l2_lambdas))

for l2 in l2_lambdas:
     cv_rss = k_folds_regression(k = 10,
                                 X = complex_train,
                                 Y = Y_train,
                                 eta = eta,
                                 tolerance = tolerance,
                                 max_iterations = 100,
                                 l2_penalty = l2)
     print('rss = %.0f\tlambda = %.0f' %(cv_rss, l2))

# in this case, the model with no l2_penalty (lambda = 0) performs best
# (smallest RSS)




# Lasso Regression: Add l1_penalty-----------------------------------------
train_data, test_data = sales.random_split(0.8, seed = 0)
all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront',
                'view',
                'condition',
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built',
                'yr_renovated']
(train_X, train_Y) = uf.get_numpy_data(train_data, all_features, 'price')

# Noramlize features:
train_X_normalized, train_norms = uf.normalize_features(train_X)
W_init = np.zeros(len(all_features) + 1) # intialize weights as 0s
tolerance = 1.0

# find optimal weights when l1_penalty is 1e7
w1e7 = reg.lasso_coordinate_descent(
     train_X_normalized, train_Y, W_init, 1e7, tolerance)

feat = ['intercept'] + all_features
for (f, w) in zip(feat, w1e7):
     print(f, w)


# NEXT: Optimize l1_penalty with k-folds cross validation
