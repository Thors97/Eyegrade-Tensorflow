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

import cv2
import numpy as np
import numpy.linalg as linalg
from scipy import ndimage
import math
import webbrowser

class TensorflowFeatureExtractor:
    """Default feature extractor.

    It assumes that images contain just one digit
    and ignores image corners.

    """
    def __init__(self, dim=28):
        self.dim = dim

    def extract(self, sample):
        filename = 'sample.jpg'
        cv2.imwrite(filename, sample._image) 
        image = self._reshape(sample)
        image = cv2.resize(image , (self.dim, self.dim)) 
        image = self.fitimage(image)  
        shiftx, shifty = self.getBestShift(image)
        shifted = self.shift(image, shiftx, shifty)
        image = shifted
        image = image.reshape( 28, 28, 1)
        image = np.expand_dims(image, axis=0)
        image = image.astype('float32')
        image /= 255
        return image

    def getBestShift(self , sample):
        cy, cx = ndimage.measurements.center_of_mass(sample)
        rows, cols = sample.shape
        shiftx = np.round(cols/2.0-cx).astype(int)
        shifty = np.round(rows/2.0-cy).astype(int)
        return shiftx,shifty

    def shift(self , image, sx, sy):
        rows,cols = image.shape
        M = np.float32([[1,0,sx],[0,1,sy]])
        shifted = cv2.warpAffine(image,M,(cols,rows))
        return shifted

    @staticmethod
    def fitimage(image):
        while np.sum(image[0]) == 0:
            image = image[1:]

        while np.sum(image[:,0]) == 0:
            image = np.delete(image,0,1)

        while np.sum(image[-1]) == 0:
            image = image[:-1]

        while np.sum(image[:,-1]) == 0:
            image = np.delete(image,-1,1)

        rows,cols = image.shape
        
        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            image = cv2.resize(image, (cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            image = cv2.resize(image, (cols, rows))

        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        image = np.lib.pad(image,(rowsPadding,colsPadding),'constant')
        return image

    @property
    def features_len(self):
        return self.dim * self.dim

    @staticmethod
    def _project_to_rectangle(sample, width, height):
        p = sample.corners
        corners_dst = np.array(
            [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]],
            dtype="float32",
        )
        h = cv2.findHomography(np.array(p, dtype="float32"), corners_dst)
        image = cv2.warpPerspective(sample.image, h[0], (width, height))
        return cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    @staticmethod
    def _reshape(sample):
        p = sample.corners
        width = int((cv2.norm(p[0, :], p[1, :]) + cv2.norm(p[2, :], p[3, :])) / 2)
        height = int((cv2.norm(p[0, :], p[2, :]) + cv2.norm(p[1, :], p[3, :])) / 2)
        return TensorflowFeatureExtractor._project_to_rectangle(sample, width, height)


class TensorflowCrossesFeatureExtractor(TensorflowFeatureExtractor):
    """Feature extractor for crosses.

    """

    def __init__(self, dim=28):
        super().__init__(dim=dim)

    def extract(self, sample):
        image = self._project_to_rectangle(sample, self.dim, self.dim)
        image = cv2.resize(image , (self.dim, self.dim)) 
        shiftx, shifty = self.getBestShift(image)
        shifted = self.shift(image, shiftx, shifty)
        image = shifted
        image = image.reshape( 28, 28, 1)
        image = np.expand_dims(image, axis=0)
        image = image.astype('float32')
        image /= 255
        return image


class FeatureExtractor:
    """Default feature extractor.

    It assumes that images contain just one digit
    and ignores image corners.

    """

    def __init__(self, dim=28):
        self.dim = dim


    def extract(self, sample):
        image = self._reshape(sample)
        image = deskew(image, self.dim)
        image = clear_boundbox(image)
        image = cv2.resize(image, (self.dim, self.dim))
        image_matrix = np.array(image, np.float32) / 255.0
        feature_vector = image_matrix.reshape(self.features_len)
        return feature_vector

    @property
    def features_len(self):
        return self.dim * self.dim

    @staticmethod
    def _project_to_rectangle(sample, width, height):
        p = sample.corners
        corners_dst = np.array(
            [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]],
            dtype="float32",
        )
        h = cv2.findHomography(np.array(p, dtype="float32"), corners_dst)
        image = cv2.warpPerspective(sample.image, h[0], (width, height))
        return cv2.threshold(image, 64, 255, cv2.THRESH_BINARY)[1]

    @staticmethod
    def _reshape(sample):
        p = sample.corners
        width = int((cv2.norm(p[0, :], p[1, :]) + cv2.norm(p[2, :], p[3, :])) / 2)
        height = int((cv2.norm(p[0, :], p[2, :]) + cv2.norm(p[1, :], p[3, :])) / 2)
        return FeatureExtractor._project_to_rectangle(sample, width, height)


class CrossesFeatureExtractor(FeatureExtractor):
    """Feature extractor for crosses.

    """

    def __init__(self, dim=28):
        super().__init__(dim=dim)

    def extract(self, sample):
        image = self._project_to_rectangle(sample, self.dim, self.dim)
        #        image = cv2.resize(image, (self.dim, self.dim))
        image_matrix = np.array(image, np.float32) / 255.0
        feature_vector = image_matrix.reshape(self.features_len)
        return feature_vector


class OpenCVExampleExtractor:
    def __init__(self, dim=20, threshold=False):
        self.dim = dim
        self.threshold = threshold
        self._corners_dst = np.array(
            [[0, 0], [dim - 1, 0], [0, dim - 1], [dim - 1, dim - 1]], dtype="float32"
        )
        self.features_len = 64

    def extract(self, sample):
        corners = np.array(sample.corners, dtype="float32")
        h = cv2.findHomography(corners, self._corners_dst)
        image = cv2.warpPerspective(sample.image, h[0], (self.dim, self.dim))
        if self.threshold:
            image = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY)[1]
        image = deskew(image, self.dim)
        feature_vector = self._preprocess_hog(image)
        return feature_vector

    def _preprocess_hog(self, image):
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin_ = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin_[:10, :10], bin_[10:, :10], bin_[:10, 10:], bin_[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [
            np.bincount(b.ravel(), m.ravel(), bin_n)
            for b, m in zip(bin_cells, mag_cells)
        ]
        hist = np.hstack(hists)
        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= linalg.norm(hist) + eps
        return np.float32(hist)


def deskew(image, dim):
    """Deskew an image.

    It improves classifier performance.
    The image must be a cv2 image.

    """
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    m = cv2.moments(image)
    if abs(m["mu02"]) < 1e-2:
        return image.copy()
    skew = m["mu11"] / m["mu02"]
    M = np.float32([[1, skew, -0.5 * dim * skew], [0, 1, 0]])
    image = cv2.warpAffine(image, M, (dim, dim), flags=affine_flags)
    return image


def clear_boundbox(image):
    """Clear the blank surrounding area of an image.

    It improves classifier performance.
    The image must be a cv2 image.

    """
    top = 0
    bot = image.shape[0]
    right = image.shape[1]
    left = 0
    it = 0
    for index, row in enumerate(image):
        if not np.all(row == 0) and it == 0:
            if index == image.shape[0] or not np.all(image[index + 1] == 0):
                top = index
                it = 1
        elif np.all(row == 0) and it == 1:
            bot = index
            break
    it = 0
    for index, col in enumerate(image.T):
        if (not np.all(col == 0)) and it == 0:
            if index + 2 >= image.shape[1] or not np.all(image.T[index + 2] == 0):
                left = index
                it = 1
        elif np.all(col == 0) and it == 1:
            right = index
            break
    cleared_image = image[top:bot, left:right]
    return cleared_image
