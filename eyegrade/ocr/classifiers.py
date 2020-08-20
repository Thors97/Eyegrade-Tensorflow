# Eyegrade: grading multiple choice questions with a webcam
# Copyright (C) 2010-2018 Rodrigo Arguello, Jesus Arias Fisteus
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
import json
import os

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model  
from . import preprocessing
from .. import utils


DEFAULT_TENSORFLOW_DIG_CLASS_FILE = "digits_classifier.h5"
DEFAULT_TENSORFLOW_CROSS_CLASS_FILE = "cross_classifier.h5"
DEFAULT_DIG_META_FILE = "digit_classifier_metadata.txt"
DEFAULT_TENSORFLOW_DIR = "tf"


class TensorflowClassifier:
    def __init__(self, num_classes, features_extractor, load_from_file=None):
        self.num_classes = num_classes
        self.features_extractor = features_extractor
        self.tf = load_model(TensorflowClassifier.resource(load_from_file))

    @property
    def features_len(self):
        return self.features_extractor.dim

    def resource(filename):
        print("Estamos cargando el modelo en el resource.")
        return utils.resource_path(os.path.join(DEFAULT_TENSORFLOW_DIR, filename))

    def classify(self, sample):
        #features = np.ndarray(shape=(1, self.features_len , self.features_len , 1), dtype="float32")
        features = self.features_extractor.extract(sample)
        prediction = self.tf.predict(features)
        return prediction

class TensorflowDigitClassifier(TensorflowClassifier):
    def __init__(
        self, features_extractor, load_from_file=None, confusion_matrix_from_file=None
    ):
        super().__init__(10, features_extractor, load_from_file=load_from_file)
 
    def classify_digit(self, sample):
        weights = self.classify(sample)
        digit = np.argmax(weights)
        return (digit, weights.reshape(10,))

class DefaultTensorflowDigitClassifier(TensorflowDigitClassifier):
    def __init__(
        self,
        load_from_file=DEFAULT_TENSORFLOW_DIG_CLASS_FILE,
        confusion_matrix_from_file=DEFAULT_DIG_META_FILE,
    ):
        super().__init__(
            preprocessing.TensorflowFeatureExtractor(),
            load_from_file=load_from_file,
        )

class TensorflowCrossClassifier(TensorflowClassifier):
    def __init__(self, features_extractor, load_from_file=None):
        super().__init__(2, features_extractor, load_from_file=load_from_file)

    def is_cross(self, sample):
        decision = np.argmax(self.classify(sample))
        return decision == 1


class DefaultTensorflowCrossesClassifier(TensorflowCrossClassifier):
    def __init__(self, load_from_file=DEFAULT_TENSORFLOW_CROSS_CLASS_FILE):
        super().__init__(
            preprocessing.TensorflowCrossesFeatureExtractor(), load_from_file=load_from_file
        )