import numpy as np
import pandas as pd
import tensorflow as tf

x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
act=None
init = tf.keras.initializers.Ones()
model = tf.keras.Sequential(
	[
		tf.keras.layers.Dense(1, activation=act, kernel_initializer=init),
		tf.keras.layers.Dense(2, activation=act, kernel_initializer=init),
		tf.keras.layers.Dense(2, activation=act, kernel_initializer=init),
		tf.keras.layers.Dense(1, activation=act, kernel_initializer=init)
	]
)
model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=0.001), loss = tf.keras.losses.MeanSquaredError(), metrics = ['mse'])

model.fit(x = x, y = y1, epochs = 50, batch_size=1)
print("Layer 0:")
print(model.layers[0].get_weights()[0])
print(model.layers[0].get_weights()[1])
print("Layer 1:")
print(model.layers[1].get_weights()[0])
print(model.layers[1].get_weights()[1])
print("Layer 2:")
print(model.layers[2].get_weights()[0])
print(model.layers[2].get_weights()[1])
print("Layer 3:")
print(model.layers[3].get_weights()[0])
print(model.layers[3].get_weights()[1])
