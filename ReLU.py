import numpy as np

''' transforma los valores negativos de la entrada en cero, mientras que los valores
positivos se mantienen sin cambios, lo que introduce no linealidad en la red y ayuda 
a aprender representaciones más útiles de los datos.'''
def ReLU(x):
    return np.maximum(0, x)