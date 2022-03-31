import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# 중간 고사, 기말 고사, 가산점 점수
x = np.array([[70, 85, 11], [71, 89, 18], [50, 80, 20], [99, 20, 10], [50, 10, 10]])
y = np.array([73, 82, 72, 57, 34])

model = Sequential()
model.add(Dense(1, input_dim=3, activation='linear'))
sgd = optimizers.SGD(learning_rate=0.0001)
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
model.fit(x, y, epochs=2000)

print(model.predict(x))