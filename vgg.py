from keras.models import Sequential 
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

import cv2, sys
import numpy as np

from config.global_parameters import frameWidth, frameHeight
from config.resources import vggWeightsPath

def create_VGG16():

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), batch_input_shape=(1, 3, 224, 224)))
    firstLayer = model.layers[-1]

    # placeholder tensor - contains the generated image
    inputImage = firstLayer.input

    # build the rest of the network
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000, activation='softmax'))

    model.load_weights(vggWeightsPath)

    model.pop()
    model.pop()
    return model

def predict(testImagePath, vggWeightsPath = vggWeightsPath):

    print "Creating the VGG model...",
    model = create_VGG16()
    print "OK."

    print "Loading test image...",
    testImage = cv2.imread(testImagePath)
    print "OK."
    print "Resizing image...",
    resizedTestImage = cv2.resize(testImage, (224, 224))
    resizedTestImage = np.reshape(resizedTestImage, (1,resizedTestImage.shape[2], resizedTestImage.shape[0], resizedTestImage.shape[1]))
    print "OK."

    print "Prediciting classes..."
    predictedClasses = model.predict(resizedTestImage)[0]
    print predictedClasses.shape
    print "OK."

    print predictedClasses

if __name__=="__main__":
    predict(sys.argv[1])

