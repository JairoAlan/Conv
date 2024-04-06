import numpy as np

# Funcion Sigmoidal
def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))