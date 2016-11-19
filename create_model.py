from glob import glob
import numpy as np

from utils import load_pkl, dump_pkl
from config.global_parameters import frameWidth, frameHeight, genreLabels 
from config.resources import video_resource
from video import extract_feature_video, gather_videos, get_frames, sequencify
from model_utils import lstm_model, get_features

from keras.utils.np_utils import to_categorical

"""testing hualos"""
from keras import callbacks
remote = callbacks.RemoteMonitor(root='http://localhost:9000')
#collect videos for each genre
#Break into sequences
#create your model
#train/fit it
#save the model
#test it


   
def train_classifier(genres=['romance', 'horror', 'action']):
    
    """Gather data for selected genres"""
    trainingData = []
    trainingLabels = []
    num_of_classes = len(genres)
    num_of_frames = 9
    print "Number of classes:",num_of_classes
    for genreIndex, genre in enumerate(genres):
        try:
            genreFeatures = load_pkl(genre+"_ultimate_vgg16")
            genreFeatures = np.array([np.array(f) for f in genreFeatures]) # numpy hack
        except Exception as e:
            print e
            return
        for videoFeatures in genreFeatures:
            for sequence in sequencify(videoFeatures, num_of_frames):
                trainingData.append(sequence)
                trainingLabels.append(genreIndex)
    num_of_samples = len(trainingLabels)

    trainingDataTensor = np.zeros((num_of_samples, num_of_frames, 4096))
    print num_of_samples
    for sampleIndex in range(num_of_samples):
        for vectorIndex in range(num_of_frames):
            try:
                trainingDataTensor[sampleIndex][vectorIndex] = trainingData[sampleIndex][vectorIndex]
            except Exception as e:
                print e
                continue
    print trainingDataTensor.shape
    model = lstm_model(num_of_classes, num_of_frames, 4096)
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    nb_epoch = 100
    batch_size = 10
    model.fit(trainingDataTensor, trainingLabels, nb_epoch=nb_epoch, batch_size=batch_size)
    model.save("data/models/lstm_"+str(num_of_classes)+"g_bs"+str(batch_size)+"_ep"+str(nb_epoch)+"nf_"+str(num_of_frames)+".h5")

if __name__=="__main__":
    train_classifier(genres=['action', 'drama', 'horror', 'romance'])
