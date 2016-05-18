import sframe
import string
import numpy as np

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


def polynomial_sframe(feature, degree):
    '''
    For a given feature vector x, create a new sframe with columns:
    x, x^2, x^3, ..., x^degree.
    
    @param feature (vector):
      the vector of the feature x
    @param degree (int):
      the highest order polynomial to be considered

    @return: poly_sframe (sframe):
      a new sframe with columns: x, x^2, x^3, ... x^degree
    '''

    # Initialize
    poly_sframe = sframe.SFrame()
    poly_sframe['x'] = feature

    if degree > 1:
        for power in range(2, degree + 1):
            name = 'x^' + str(power)
            poly_sframe[name] = feature.apply(lambda x: x ** power)

    return poly_sframe

# Test
#test_vec = sframe.SArray([1, 2, 3, 4, 5])
#test_sf = polynomial_sframe(test_vec, 5)
#print test_sf



def normalize_features(X):
    '''
    Normalize the columns of a model matrix. 
    NOTE: this is the value as a fraction of its vector norm and NOT standard
      deviates (Z scores)

    @param X (matrix):
      the model matrix
    @return list:
      normalized_features: the normalized model matrix (matrix)
      norms: the norms of each column (used to repeat same normalization on 
        test data (vector)
    '''
    norms = np.linalg.norm(X, axis = 0)
    normalized_features = X / norms

    return (normalized_features, norms)


def strip_punctuation(text):
    return text.translate(None, string.punctuation)
