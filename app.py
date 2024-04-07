from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from Conv import Conv
from ReLU import ReLU
from Pool import Pool
from Softmax import Softmax
from TestMnistConv import W1,W5,Wo
import os


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener la imagen cargada por el usuario
    image = request.files['image']
    img = Image.open(image).convert('L')  # Convertir a escala de grises
    img = img.resize((28, 28))  # Redimensionar a 28x28 (tamaño MNIST)

    # Convertir la imagen a un array NumPy y normalizar
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255
    img_array = 1 - img_array  # Invertir los colores (MNIST es fondo negro con dígitos blancos)

    # Realizar la predicción utilizando la CNN
    y1 = Conv(img_array, W1)
    y2 = ReLU(y1)
    y3 = Pool(y2)
    y4 = np.reshape(y3, (-1, 1))
    v5 = np.matmul(W5, y4)
    y5 = ReLU(v5)
    v = np.matmul(Wo, y5)
    y = Softmax(v)

    # Obtener el número predicho
    predicted_number = np.argmax(y)

    # Devolver el resultado de la predicción como JSON
    return jsonify({'Numero Predicho': int(predicted_number)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
