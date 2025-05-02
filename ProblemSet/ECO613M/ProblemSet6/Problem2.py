import pandas as pd
import numpy as np
import sklearn.neural_network as sknn
import tensorflow as tf
import keras
np.random.seed(42)

df = pd.read_csv("ProblemSet/ECO613M/ProblemSet6/spambase.data", header=None)
X = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

N = len(X)
k = N//25
x_train = X[k:]
y_train = y[k:]
x_test = X[:k]
y_test = y[:k]

model = sknn.MLPClassifier(hidden_layer_sizes = 50, activation='relu', max_iter = 1000, solver = 'adam')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(np.mean((y_test - y_pred) ** 2))

#Tensorflow Model 

input_layer = tf.keras.layers.Input(shape = [x_train.shape[1]], name = 'Input Layer')
hidden1 = tf.keras.layers.Dense(40, activation = 'relu', name = 'hidden1')(input_layer)
# hidden2 = tf.keras.layers.Dense(512, activation = 'relu', name = 'hidden2')(hidden1)
# hidden3 = tf.keras.layers.Dense(64, activation = 'relu', name = 'hidden3')(hidden2)
output_layer = tf.keras.layers.Dense(units = 1, activation = 'sigmoid')(hidden1)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x_train, y_train, epochs=50)

y_pred = model.predict(x_test)
y_predict = (y_pred > 0.5).astype(int).reshape(-1)
print(np.mean(y_predict != y_test))
