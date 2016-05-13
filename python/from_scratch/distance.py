# Different distance metrics
import numpy as np


def dist_euclidean(v1, v2):
    '''
    Calculate the Euclidean distance between 2 points

    @params v1, v2 (vectors):
      the coordinates of point 1 and point 2

    @return dist (float):
      the Euclidean distance
    '''
    v1 = np.array(v1)
    v2 = np.array(v2)
    sq_diffs = (v1 - v2) ** 2
    return np.sqrt(sum(sq_diffs))
    


# Test
#v1 = [1, 1, 1]
#v2 = [2, 2, 2]

#print dist_euclidean2(v1, v2)
