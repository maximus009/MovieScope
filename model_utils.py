from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import LSTM
from keras.layers.convolutional import Convolution2D, Convolution3D, MaxPooling2D
from keras import backend
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

def remove_last_layers(model):
    """To remove the last FC layers of VGG and get the 4096 dim features"""
    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []


vgg_model_16 = VGG16(include_top=True, weights="imagenet")
vgg_model_19 = VGG19(include_top=True, weights="imagenet")

remove_last_layers(vgg_model_16)
remove_last_layers(vgg_model_19)


def get_features_batch(frames, model_name="vgg16"):

    if model_name.lower() in ["vgg16", "vgg_16"]:
        model = vgg_model_16

    if model_name.lower() in ["vgg19", "vgg_19"]:
        model = vgg_model_19

    imageTensor = np.array(frames)

    ### /255 causing error. Maybe Vanishing gradients
    modelFeature =  model.predict(imageTensor, verbose=1)
    return modelFeature



def get_features(image, model_name="vgg16"):

    if backend.image_dim_ordering()=='th':
        print "Please switch to tensorflow backend. Update to reorder will come soon."
        return None

    if model_name.lower() in ["vgg16", "vgg_16"]:
        model = vgg_model_16

    if model_name.lower() in ["vgg19", "vgg_19"]:
        model = vgg_model_19

    imageTensor = np.zeros((1, 224, 224, 3))
    imageTensor[0] = image

    ### /255 causing error. Maybe Vanishing gradients
    modelFeature =  model.predict(imageTensor)[0]
    return modelFeature

def spatial_model(number_of_classes=2):
    """Classification layers here."""

    model = Sequential()
    model.add(Dense(2048, input_dim=4096, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(number_of_classes, activation='softmax'))

    return model


def lstm_model(number_of_classes=2, number_of_frames=None, input_dim=4096):
    """Classification layers here with LSTM."""

    if number_of_frames == None:
        print  "Need to specify the number of frames (as timestep)."
        return
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, stateful=False, input_shape=(number_of_frames, input_dim)))
    model.add(LSTM(64, return_sequences=True, stateful=False))
    model.add(LSTM(64))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))

    return model


def optical_flow_model(number_of_classes=2):

    model = Sequential()
    model.add(Convolution2D(48, 7, 7, border_mode='same', input_shape=(4, 224, 224), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(number_of_classes, activation='softmax'))

    return model

if __name__=="__main__":
    import cv2
    inputImage = cv2.resize(cv2.imread("testImages/test1.jpg"), (224, 224))
    from time import time
    start = time()
    vector = get_features(inputImage, 'vgg19')
    print 'time taken by vgg 19:',time()-start,'seconds. Vector shape:',vector.shape
    start = time()
    vector = get_features(inputImage, 'vgg16')
    print 'time taken by vgg 16:',time()-start,'seconds. Vector shape:',vector.shape
