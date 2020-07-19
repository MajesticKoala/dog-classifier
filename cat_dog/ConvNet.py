import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import time

#Set up TensorBoard
log_path = f'logs/cats-vs-dogs-cnn-64x2-{int(time.time())}'
tensorboard = TensorBoard(log_dir=log_path, histogram_freq=1)

X = np.load('features.npy')
y = np.load('labels.npy')

#Example image
#print(y[515])
#plt.imshow(X[515].reshape(60,60), cmap='gray')
#plt.show()

X = X/255.0 #Normalize the imageto value between 0-1

#Create layers to neural net
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
#model.add(Dense(64))
#model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=['accuracy'])

#Train model
model.fit(X, y, batch_size=12, epochs=3, validation_split=0.1, callbacks=[tensorboard])
