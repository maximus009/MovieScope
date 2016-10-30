from glob import glob
import numpy as np

from utils import load_pkl, dump_pkl
from config.global_parameters import frameWidth, frameHeight
from config.resources import video_resource

from video import extract_feature_video 
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


def gather_videos(genre, limit_videos = 25):
    videoPaths = glob(video_resource+genre+'/*')
    videoFeatures = np.array([list(extract_feature_video(videoPath, verbose=True)) for videoPath in videoPaths[:limit_videos]])
    return videoFeatures


def gather():
    print "Gathering features for Horror...",
    horrorFeatures = gather_videos('horror')
    print horrorFeatures.shape
    print "OK"
    print "Gathering features for Romance...",
    romanceFeatures = gather_videos('romance')
    print romanceFeatures.shape
    print "OK."
    print "Dumping...",
    dump(horrorFeatures, open('data/horror.p','wb'))
    dump(romanceFeatures, open('data/romance.p','wb'))
    print "OK"


def train():
    romanceFeatures = load_pkl('romance')
    horrorFeatures = load_pkl('horror')
    romanceFeatures = np.array([np.array(f) for f in romanceFeatures])
    horrorFeatures = np.array([np.array(f) for f in horrorFeatures])

    model = spatial_model(2)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    trainingData = []
    trainingLabels = []
    num_of_frames = 35
    for i,videoFeatures in enumerate(romanceFeatures):
        randomIndices = sorted(np.random.randint(0,len(videoFeatures),num_of_frames))
        selectedFeatures = np.array(videoFeatures[randomIndices])
        for feature in selectedFeatures:
            trainingData.append(feature)
            trainingLabels.append([1,0])

    for i,videoFeatures in enumerate(horrorFeatures):
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

    
if __name__=="__main__":
    train()

