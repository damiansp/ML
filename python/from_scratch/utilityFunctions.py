import sframe


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
