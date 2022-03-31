from tkinter import E
import numpy as np

from tensorflow.keras.layers import Dense
import tensorflow as tf


x = np.array([0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0])
y = np.array([0, 0, 0, 1, 1, 1])

model = tf.keras.Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(x, y, epochs=2000)
print(model.predict(x))