import numpy as np
''' Toma un vector de valores como entrada y produce un vector 
de la misma longitud donde cada valor representa la probabilidad 
de que la entrada pertenezca a una de las clases distintas. '''

def Softmax(x):
    x  = np.subtract(x, np.max(x))        # prevent overflow
    ex = np.exp(x)
    
    return ex / np.sum(ex)