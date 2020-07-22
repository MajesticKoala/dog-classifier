from flask import Flask, render_template, request, redirect
from testModel import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from PIL import Image

app = Flask(__name__)

model = load_model()

IMG_SIZE = 70

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.get_data()

@app.route('/upload', methods=['POST'])
def add_image():
    if request.files:
        in_image = request.files["image"]
        image = in_image.read()
        image_name = in_image.name()

        print(image_name, file=sys.stderr)

        npimg = np.fromstring(image, np.uint8)
        img_array = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        big_array = cv2.resize(img_array, (200, 200))
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

        img_print = Image.fromarray(big_array)
        img_predict = Image.fromarray(new_array)
        img_predict.save('images/', image_name, '.jpg')

        prepared_img=new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        prediction=model.predict(['images/', image_name, '.jpg'])

    return render_template('index.html', predict=prediction )

if __name__ == "__main__":
    app.run(debug=True)