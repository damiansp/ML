# Different distance metrics
import numpy as np


def dist_euclidian(v1, v2):
    '''
    Calculate the Euclidian distance between 2 points

    @params v1, v2 (vectors):
      the coordinates of point 1 and point 2

    @return dist (float):
      the Euclidian distance
    '''
    v1 = np.array(v1)
    v2 = np.array(v2)
    sq_diffs = (v1 - v2) ** 2
    return np.sqrt(sum(sq_diffs)) 

# Test
#v1 = [1, 1, 1]
#v2 = [2, 2, 2]

#print dist_euclidean2(v1, v2)   

def get_all_distances_euclidian(X, query_v):
    '''                                                                        
    For a point (record or row in X), get the distance from it to all other    
    points in X.                                                               
                                                                               
    @param X (matrix):                                                         
      the model matrix                                                         
    @param query_v (vector):                                                   
      the row (record) from which distances are being computed                 

    @return dists (vector):
      the Euclidian distance to all other rows
    '''

    return np.sqrt(np.sum((X[:] - query_v) ** 2, axis = 1))
    

# Test
#M = np.array([[0, 0, 1],
#              [0, 1, 0],
#              [0, 1, 1],
#              [1, 0, 0],
#              [1, 0, 1],
#              [1, 1, 0],
#              [1, 1, 1]])

#print get_all_distances_euclidian(M, np.array([0, 0, 0]))




