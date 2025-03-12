import numpy as np
def bic(k, n, ll):
    return -2 * ll + k * np.log(n)