from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras import backend
import numpy as np

vgg_model_16 = VGG16(include_top=True, weights="imagenet")
vgg_model_19 = VGG19(include_top=False, weights="imagenet")

def get_features(image, model_name="vgg16"):

    if backend.image_dim_ordering()=='th':
        print "Please switch to tensorflow backend. Update to reorder will come soon."
        return None

    if model_name in ["vgg16", "VGG16"]:
        model = vgg_model_16
    if model_name in ["vgg19", "VGG19"]:
        model = vgg_model_19

    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    imageTensor = np.zeros((1, 224, 224, 3))
    imageTensor[0] = image

    modelFeature =  model.predict(imageTensor)[0]
    return modelFeature


if __name__=="__main__":
    import cv2
    vector = get_features(cv2.resize(cv2.imread("testImages/test1.jpg"), (224,224)))
    print vector.shape
