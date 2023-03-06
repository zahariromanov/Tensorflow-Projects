# init setup for Python DL Projects
import tensorflow as tf
import numpy as np
import pandas as pd

print('TensorFlow version: ' , tf.__version__)
print('NumPy version: ' , np.__version__)
print('Pandas version: ' , pd.__version__)

# import tensorflow_datasets as tfds
## Construct a tf.data.Dataset
# ds = tfds.load('mnist', split='train', shuffle_files=True)
#
## Build your input pipeline
# ds = ds.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
# for example in ds.take(1):
#   image, label = example["image"], example["label"]
 
 
# Hello World! 
print("Hello World!")

name = (input("What is your name? \n : "))
print("Hello", name, "! I hope we build something great together! \n\n\n")

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])

print(predictions, "\n\n\n")



# view results
#
#
# TensorFlow version:  2.11.0
# Hello World!       
# What is your name? 
#  : Zak
# Hello Zak ! I hope we build something great together! 
#
#
#
# 2023-03-02 11:59:50.384049: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Epoch 1/5
# 1875/1875 [==============================] - 2s 1ms/step - loss: 0.3047 - accuracy: 0.9115
# Epoch 2/5
# 1875/1875 [==============================] - 2s 1ms/step - loss: 0.1472 - accuracy: 0.9560
# Epoch 3/5
# 1875/1875 [==============================] - 2s 1ms/step - loss: 0.1104 - accuracy: 0.9668
# Epoch 4/5
# 1875/1875 [==============================] - 2s 1ms/step - loss: 0.0897 - accuracy: 0.9724
# Epoch 5/5
# 1875/1875 [==============================] - 2s 1ms/step - loss: 0.0771 - accuracy: 0.9760
# 313/313 - 0s - loss: 0.0718 - accuracy: 0.9775 - 277ms/epoch - 884us/step
# [[-0.1539109   0.08710244 -0.22000483 -0.00269127 -0.5885928   0.04445815
#   -0.6053179  -0.3560646  -0.14792997 -0.46962225]]
#
#
#
# end results 