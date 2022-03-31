from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

import tensorflow as tf

x = [1, 2 ,3, 4, 5, 6, 7, 8, 9]
y = [11, 22, 10, 32, 98, 35, 77, 87, 2]

model = Sequential()

# 출력 y의 차원은 1, 입력 x의 차원은 1
# 선형 회귀이므로 activation은 'linear'
model.add(keras.layers.Dense(1, input_dim=1, activation='linear'))

# sgd는 경사 하강법을 의미, 학습률(learning rate, lr)은 0.01
sgd = optimizers.SGD(learning_rate=0.01)

# 손실 함수(Loss function)은 평균제곱오차 mse를 사용
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])

# 주어진 x와 y데이터에 대해서 오차를 최소화하는 작업을 300번 시도합니다.
model.fit(x, y, epochs=300)

plt.plot(x, model.predict(x), 'b', x, y, 'k.')
plt.show()
