import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.datasets import mnist
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split


# mnist = tf.keras.datasets.mnist
# Load data
print ("TEMKENG")
((train_data, train_labels), (eval_data, eval_labels)) = mnist.load_data()
# 80%
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2,random_state=1)
print(train_data.shape,train_labels.shape)
print(val_data.shape, val_labels.shape)

# reg_model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(6, 5, activation='relu', padding='same'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(16, 5, activation='relu', padding='same'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(120, activation='relu'),
#     tf.keras.layers.Dense(84, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax'),
#
# ])
#
# reg_model.summary()
# train_img = Image.fromarray(train_data[0])
# test_img = Image.fromarray(val_data[0])
# train_img.save("train_image.png")
# test_img.save("test_image.png")
#
regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)
# Convolutional Layer #1
model = models.Sequential()
model.add(layers.Conv2D(6, (5, 5),padding="same", activation=tf.nn.relu, input_shape=(28, 28, 1), kernel_regularizer=regularizer))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (5, 5), activation='relu', kernel_regularizer=regularizer))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(120,activation='relu'))
model.add(layers.Dense(84,activation='relu'))
model.add(layers.Dense(10))

# model.summary()
# minibatch_placeholder = tf.placeholder(tf.float32, (128,28*28))
# print("aaaaaaaaa")
# reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# print(reg)
# regularization_losses = tf.add_n(reg)

# predictions = layers.Dense(10,activation='softmax', name='x_train_out')(train_data)
# conv1 = tf.keras.layers.Conv2D(train_data,6, (5,5), padding="same", activation=tf.nn.relu)
# conv1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# print(test_data.shape)
# print(train_data.shape)
# print(type(conv1))
# conv1 = tf.layers.conv2d(inputs=train_data, filters=6, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
# Pooling Layer #1
# pool1 = tf.keras.layers.MaxPool2D(inputs=conv1, pool_size=[2, 2], strides=2)
# print(conv1.output_shape)


