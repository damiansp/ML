import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '../'))

import distance


def get_knn(k, X, query_v):
    '''
    For a row in X, find the k nearest neighbors as defined by a given 
    distance metric
    
    TO DO: Currently only supports Euclidian distance.  Update to use
    an arbitrary distance metric.

    @param k (int):
      the number of nearest neighbors to find
    @param X (matrix):
      the model matrix
    @param query_v (vector):
      the row in X for which the nearest neigbors are being sought

    @return:
      the indices for the nearest neigbors (sorted by distance)
    '''
    dists = distance.get_all_distances_euclidian(X, query_v)
    sorted_indices = np.argsort(dists)
    return sorted_indices[:k]

def knn_predict(k, X, Y, query_v):
    '''
    Predicts the Y value of an input vector by averaging the Y values of the 
    k nearest neighbors.
    
    @param k (int):                                                            
      the number of nearest neighbors to base the estimate on
    @param X (matrix):                                                         
      the model matrix                                                         
    @param Y (vector):
      the response variable
    @param query_v (vector):                                                   
      the data for which the prediction is being sought (must be ordered as a
      row in X)
    @return (float):
      the estimated value of the response variable for the query_v

    TO DO: instead of simply returning the mean, allow a kernel estimator
    '''
    nn = get_knn(k, X, query_v)
    response_nn = Y[nn]
    return np.mean(response_nn)

def knn_predict_all(k, X, Y, query_vs):
    '''                                                                        
    Predicts the Y value of an input vector by averaging the Y values of the   
    k nearest neighbors.                                                      
                                                                               
    @param k (int):                                                           
      the number of nearest neighbors to base the estimate on                  
    @param X (matrix):                                                         
      the model matrix                                                         
    @param Y (vector):                                                         
      the response variable                                                    
    @param query_vs (matrix):                                                  
      the data for which the predictions are being sought 
      (must be ordered as X)

    @return predictions (vector):
      the estimated values of the response variable for each of the query_vs 
    '''
    predictions = []

    for qv in query_vs:
        pred = knn_predict(k, X, Y, qv)
        predictions.append(pred)

    return predictions

