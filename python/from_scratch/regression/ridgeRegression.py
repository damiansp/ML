import sframe
import numpy as np
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '../'))
import utilityFunctions as uf

sales = sframe.SFrame('../../../data/kc_house_data.csv')
sales = sales.sort(['sqft_living', 'price'])

# Split data into train/validation, and test sets
(train_valid, test) = sales.random_spit(0.9, seed = 1)
train_valid_shuffled = 
