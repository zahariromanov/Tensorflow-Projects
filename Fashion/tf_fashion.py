# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Download the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Explore the data
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images.shape
len(train_labels)
train_labels
test_images.shape
len(test_labels)

# Preprocess the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adamax',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# Make predictions
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

predictions[0]
np.argmax(predictions[0])
test_labels[0]

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Verify predictions
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Grab an image from the test dataset. Use the model to predict the image's label.
# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)
predictions_single = probability_model.predict(img)

print(predictions_single)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
np.argmax(predictions_single[0])

# Trained with Ryzen 9 5950X, 64GB RAM, RTX 3060 Ventus 3x OC, Windows 11 Pro, Python 3.9.7, TensorFlow
#
# results:
#
# 2.11.0
# 2023-03-05 23:00:31.381273: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Epoch 1/10
# 1875/1875 [==============================] - 2s 990us/step - loss: 0.4952 - accuracy: 0.8247
# Epoch 2/10
# 1875/1875 [==============================] - 2s 907us/step - loss: 0.3758 - accuracy: 0.8652
# Epoch 3/10
# 1875/1875 [==============================] - 2s 900us/step - loss: 0.3358 - accuracy: 0.8780
# Epoch 4/10
# 1875/1875 [==============================] - 2s 891us/step - loss: 0.3136 - accuracy: 0.8858
# Epoch 5/10
# 1875/1875 [==============================] - 2s 890us/step - loss: 0.2961 - accuracy: 0.8913
# Epoch 6/10
# 1875/1875 [==============================] - 2s 886us/step - loss: 0.2806 - accuracy: 0.8948
# Epoch 7/10
# 1875/1875 [==============================] - 2s 886us/step - loss: 0.2691 - accuracy: 0.9001
# Epoch 8/10
# 1875/1875 [==============================] - 2s 883us/step - loss: 0.2584 - accuracy: 0.9045
# Epoch 9/10
# 1875/1875 [==============================] - 2s 877us/step - loss: 0.2470 - accuracy: 0.9076
# Epoch 10/10
# 1875/1875 [==============================] - 2s 883us/step - loss: 0.2394 - accuracy: 0.9113
# 313/313 - 0s - loss: 0.3655 - accuracy: 0.8747 - 238ms/epoch - 760us/step
# 
# Test accuracy: 0.8747000098228455
# 313/313 [==============================] - 0s 496us/step
# (28, 28)
# (1, 28, 28)
# 1/1 [==============================] - 0s 11ms/step
# [[4.7043734e-05 3.4456119e-13 9.9819142e-01 1.5880787e-10 3.6941661e-04
#   5.8762386e-13 1.3921025e-03 1.3269940e-20 3.4339059e-10 1.0994147e-15]]
# 
#
#
#
#