import numpy as np

def reverse_p(x):
    return -np.log(1/x - 1)

def transform_p(x):
    return 1/(1+np.exp(x))