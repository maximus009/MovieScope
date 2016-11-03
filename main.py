from glob import glob
import numpy as np

from utils import load_pkl, dump_pkl
from config.global_parameters import frameWidth, frameHeight
from config.resources import video_resource

from video import extract_feature_video, gather_videos
from model_utils import spatial_model 

"""testing hualos"""
from keras import callbacks
remote = callbacks.RemoteMonitor(root='http://localhost:9000')
#collect videos for each genre
#extract spatial features for each
#create your model
#train/fit it
#save the model
#test it



def gather_genre(genre, limit_videos=100):

    print "Gathering features for",genre,
    genreFeatures = gather_videos(genre, limit_videos)
    print "OK."
    print genreFeatures.shape
    dump_pkl(genreFeatures, genre+str(limit_videos))


def gather():
    gather_genre('action')


def train():
    romanceFeatures = load_pkl('romance100')
    horrorFeatures = load_pkl('horror100')
    romanceFeatures = np.array([np.array(f) for f in romanceFeatures])
    horrorFeatures = np.array([np.array(f) for f in horrorFeatures])

    model = spatial_model(2)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    trainingData = []
    trainingLabels = []
    num_of_frames = 35
    for videoFeatures in romanceFeatures:
        if len(videoFeatures) > num_of_frames:
            randomIndices = sorted(np.random.randint(0,len(videoFeatures),num_of_frames))
            selectedFeatures = np.array(videoFeatures[randomIndices])
            for feature in selectedFeatures:
                trainingData.append(feature)
                trainingLabels.append([1,0])

    for videoFeatures in horrorFeatures:
        if len(videoFeatures) > num_of_frames:
            randomIndices = sorted(np.random.randint(0,len(videoFeatures),num_of_frames))
            selectedFeatures = np.array(videoFeatures[randomIndices])
            for feature in selectedFeatures:
                trainingData.append(feature)
                trainingLabels.append([0,1])


    trainingData = np.array(trainingData)
    trainingLabels = np.array(trainingLabels)
    trainingLabels = trainingLabels.reshape((-1,2))
    print trainingData.shape
    print trainingData[0].shape
    print trainingLabels.shape

    model.fit(trainingData, trainingLabels, batch_size=16, nb_epoch=50, callbacks=[remote])

    model.save("all_bs_16_ep_50_nf_35.h5")

    
if __name__=="__main__":
    gather()
    #train()

