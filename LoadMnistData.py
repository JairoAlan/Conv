
from struct import unpack
import gzip
from numpy import uint8, zeros, float32


# Lee las imagenes de entrada y las etiquetas(0-9)
# Y la regresa como una lista de tuplas
#
def LoadMnistData(imagefile, labelfile):
    # Abre las imagines con gzip y lo lee en modo binario
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Se lee el dato binario 
    images.read(4)  # Numero magico bytes
    # Lee el encabezado de cada archivo para obtener información 
    # sobre el número de imágenes, el tamaño de las imágenes y el número de etiquetas.
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Se lee el el metadato de las etiquetas
    labels.read(4)  # numero magico
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('numeros de las etiquetas no coinciden con el numero de imagenes')

    # obtiene la data
    x = zeros((N, rows, cols), dtype=float32)  # Inicializa numpy array
    y = zeros((N, 1), dtype=uint8)  # Inicializa numpy array
    # Itera sobre las imágenes y lee los píxeles de cada imagen. 
    # Los píxeles se almacenan en la matriz x.
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)
            
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Solo un byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel

        tmp_label = labels.read(1)
        # Lee las etiquetas correspondientes a cada imagen y las almacena en la matriz y.
        y[i] = unpack('>B', tmp_label)[0]
    # Devuelve una tupla que contiene la matriz de imágenes x y la matriz de etiquetas y.
    return (x, y)