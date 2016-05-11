import matplotlib.pyplot as plt
import numpy as np
import sframe
import os, sys
import regression as reg

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                          '../'))
import utilityFunctions as uf



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





# Ridge Regression: Add l2_penalty
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

