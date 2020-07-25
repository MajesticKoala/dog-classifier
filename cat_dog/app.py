from flask import Flask, render_template, request, redirect, flash
from testModel import load_model, prepare
import numpy as np
import matplotlib.pyplot as plt
import cv2, sys, os, shutil
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

IMG_SIZE = 70
UPLOAD_FOLDER = 'images/'

@app.route('/')
def index():
    return render_template('index.html')

def predict(image_array):
    CATEGORIES = ['Dog 🐶', 'Cat 🐱']

    model = tf.keras.models.load_model("64x3-CNN-70px.model")
    prediction = model.predict([image_array])
    return CATEGORIES[int(prediction[0][0])]

def clear_directory():
    folder = 'images'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e), file=sys.stderr)

@app.route('/upload', methods=['POST'])
def add_image():
    #First clear image directory
    clear_directory()
    if request.files:
        in_image = request.files["image"]
        image = in_image.read()
        image_name = in_image.filename

        print(image_name, file=sys.stderr)

        if image_name == '':
            return redirect('/')

        #Create image arrays
        npimg = np.fromstring(image, np.uint8)
        img_array = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        big_array = cv2.resize(img_array, (200, 200))
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

        #Create images from arrays and save in images folder
        img_print = Image.fromarray(big_array)
        img_predict = Image.fromarray(new_array)
        img_predict.save(os.path.join(UPLOAD_FOLDER, image_name))
        prepared_img=new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        prediction = predict(prepared_img)

    return render_template('index.html', predict=prediction, file_name=image_name )

if __name__ == "__main__":
    app.run(debug=True)