# Eyegrade: grading multiple choice questions with a webcam
# Copyright (C) 2010-2018 Jesus Arias Fisteus
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <https://www.gnu.org/licenses/>.
#
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model

import random
import argparse
import sys
import cv2
import os
import numpy as np
import math
from scipy import ndimage
import seaborn as sn
from sklearn.model_selection import train_test_split


def getBestShift(sample):
    cy, cx = ndimage.measurements.center_of_mass(sample)
    rows, cols = sample.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty


def shift(image, sx, sy):
    rows, cols = image.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(image, M, (cols, rows))
    return shifted


def fitimage(image):
    while np.sum(image[0]) == 0:
        image = image[1:]

    while np.sum(image[:, 0]) == 0:
        image = np.delete(image, 0, 1)

    while np.sum(image[-1]) == 0:
        image = image[:-1]

    while np.sum(image[:, -1]) == 0:
        image = np.delete(image, -1, 1)

    rows, cols = image.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        image = cv2.resize(image, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        image = cv2.resize(image, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2.0)),int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)),int(math.floor((28 - rows) / 2.0)))
    image = np.lib.pad(image, (rowsPadding, colsPadding), 'constant')
    return image


# Con esta función cargamos desde un directorio las imagenes de Eyegrade para realizar el entrenamiento.
def load_images(path):
    x_test = []
    y_test = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(255 - img, (28, 28))
        (thresh, img) = cv2.threshold(img, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img = fitimage(img)
        shiftx, shifty = getBestShift(img)
        shifted = shift(img, shiftx, shifty)
        img = shifted
        if img is not None:
            x_test.append(img)
            y_test.append(int(filename[7]))
    return x_test, y_test


def mix_Samples(path):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    images, labels = load_images(path)
    x_trainE, x_testE, y_trainE, y_testE = train_test_split(images, labels, test_size=0.5, shuffle=True)
    x_testE = np.array(x_testE)
    x_trainE = np.array(x_trainE)
    y_trainE = np.array(y_trainE)
    y_testE = np.array(y_testE)
    x_train = np.append(x_train, x_trainE, axis=0)
    x_test = np.append(x_test, x_testE, axis=0)
    y_train = np.append(y_train, y_trainE, axis=0)
    y_test = np.append(y_test, y_testE, axis=0)
    return (x_train, y_train), (x_test, y_test)


def create_classifier(num_classes, x_train, x_test, y_train, y_test):
    batch_size = 128
    epochs = 1
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    model = tf.keras.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation=tf.nn.relu, input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation=tf.nn.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation=tf.nn.relu))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=tf.nn.softmax))
    optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer=optimizer, metrics=['accuracy'])
    print(x_train.shape , x_test.shape)
    hist = model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    return model

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Create the Tensorflow classifier for digits or crosses."
    )
    parser.add_argument(
        "path", help='path whit the samples for training/test'
    )
    parser.add_argument(
        "classifier", help='classifier to be created ("digits" or "crosses")'
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    (x_train, y_train), (x_test, y_test) = mix_Samples(args.path)

    if args.classifier == "digits":
        model = create_classifier(10, x_train, x_test, y_train, y_test)
        model_name = 'digitsClassifier_' + str(random.randint(0, 100000000000000)) + '.h5'
    else:
        model = create_classifier(2, x_train, x_test, y_train, y_test)
        model_name = 'crossesClassifier_' + str(random.randint(0, 100000000000000)) + '.h5'
    model.save('..\\data\\tf\\' +  model_name)
    print(model_name)


if __name__ == "__main__":
    main()
