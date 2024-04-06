import numpy as np
from scipy import signal
from LoadMnistData import *
from Softmax import *
from ReLU import *
from Conv import *
from Pool import *
from MnistConv import *


# Learn
# Carga  los datos de entrenamiento y prueba
Images, Labels = LoadMnistData('MNIST\\t10k-images-idx3-ubyte.gz', 'MNIST\\t10k-labels-idx1-ubyte.gz')
Images = np.divide(Images, 255) # Normalizacion dividiendo las imagenes por 255

# Inicia los pesos de las Capas Convolucionales
W1 = 1e-2 * np.random.randn(9, 9, 20)
W5 = np.random.uniform(-1, 1, (100, 2000)) * np.sqrt(6) / np.sqrt(360 + 2000)
Wo = np.random.uniform(-1, 1, ( 10,  100)) * np.sqrt(6) / np.sqrt( 10 +  100)

# Divide el conjunto de datos en imágenes de entrenamiento (X) y etiquetas (D).
X = Images[0:8000, :, :]
D = Labels[0:8000]

# Itera a lo largo de un número específico de épocas (en este caso, 5) para entrenar la red.    
for _epoch in range(5):
    print(_epoch)
    # En cada época, llama a la función MnistConv que realiza el entrenamiento de la red con los datos de entrada y las etiquetas.
    W1, W5, Wo = MnistConv(W1, W5, Wo, X, D)

    
# Test
# Utiliza las imágenes restantes y etiquetas para realizar pruebas.
X = Images[8000:10000, :, :]
D = Labels[8000:10000]

# Calcula la precisión (acc) de la red en el conjunto de prueba.
acc = 0
N   = len(D)
# Itera sobre cada imagen de prueba, pasándola a través de la red neuronal convolucional.
for k  in range(N):
    x  = X[k, :, :]
    # AQUI
    y1 = Conv(x, W1)
    y2 = ReLU(y1)
    y3 = Pool(y2)
    y4 = np.reshape(y3, (-1, 1))
    v5 = np.matmul(W5, y4)
    y5 = ReLU(v5)
    v  = np.matmul(Wo, y5)
    # Utiliza la función Softmax para obtener la salida de la red.
    y  = Softmax(v)
    
    i = np.argmax(y)
    # Compara la salida con la etiqueta real para calcular la precisión.
    if i == D[k][0]:
        acc = acc + 1
        
acc = acc / N
print("Exactitud : ", acc)


