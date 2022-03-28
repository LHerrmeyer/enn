#!/usr/bin/env python.exe
import tensorflow as tf
import numpy as np
import pandas as df
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X = data.data # cols: Sepal length, sepal width, petal length, petal width
y = data.target # Setosa (0), versicolor (1), virginica (2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)

model = tf.keras.models.Sequential([
	tf.keras.layers.Dense(4, activation=tf.nn.relu),
	tf.keras.layers.Dense(4, activation=tf.nn.relu),
	tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])
model.compile(optimizer="adam",
				loss="sparse_categorical_crossentropy",
				metrics=["accuracy"])

model.fit(X_train, y_train, epochs=1000, validation_split=0.3, verbose=0)

print("Accuracy:")
model.evaluate(X_test, y_test)

for i in range(0, 3):
	weights = model.layers[i].get_weights()
	shape = np.shape(weights[0])
	print(f"Layer {i} {shape[1]}x{shape[0]}")
	print(np.transpose(weights[0]))
	print(np.transpose(weights[1]))

np.set_printoptions(suppress=True)
test_set = X_test[0:5]
preds = model.predict(test_set)
print("Test set (Sepal length, sepal width, petal length, petal width):")
print(test_set)
print("Preds (setosa, versicolor, virginica):")
print(preds)
print("Preds (max prob):")
print(preds.argmax(1))
print("True results:")
print(y_test[0:5])
