import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

CATEGORIES = ['Dog', 'Cat']

def prepare(filePath):
    IMG_SIZE = 70
    img_array = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
    
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    #plt.imshow(new_array, cmap='gray')
    #plt.show()

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def load_model():
    model = tf.keras.models.load_model("64x3-CNN-70px.model")
    return model

#model = load_model()
#prediction = model.predict([prepare('data/rusty.jpg')])

#print(CATEGORIES[int(prediction[0][0])])
