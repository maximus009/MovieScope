""" Function to compute the rsults with Histogram and some classifier"""
import cv2
import os
import numpy as np
from config.resources import video_resource
from glob import glob
from video import get_frames
from utils import dump_pkl, load_pkl
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.svm import SVC


def get_histogram(image):
    color = ('b','g','r')
    feature = []
    for channel,col in enumerate(color):
        hist = cv2.calcHist([image],[channel],None,[256],[0,256])
        feature.append(hist)
    feature = np.reshape(feature, (3*256,))
    return feature


def gather_histogram_data(genre, mode='train'):
    """Driver function to collect frame features for a genre"""

    trainPath = os.path.join(video_resource,mode,genre)
    print trainPath
    videoPaths = glob(trainPath+'/*')
    genreFeatures = []
    for videoPath in videoPaths:
        print videoPath,":",
        frames =list(get_frames(videoPath, time_step=1000))
        print len(frames),
        if len(frames)==0:
            print "corrupt."
            continue
        videoFeatures = np.array([get_histogram(frame) for frame in frames])
        print videoFeatures.shape
        genreFeatures.append(videoFeatures)

    outPath = genre+"_histogram_"+mode
    dump_pkl(genreFeatures, outPath)


def train_mode(genres = ['action','drama','horror','romance']):
    trainingData, trainingLabels = [], []
    for genreIndex, genre in enumerate(genres):
        try:
            genreFeatures = np.array(load_pkl(genre+'_histogram_train'))
        except Exception as e:
            print e
            return
        for videoFeatures in genreFeatures:
            for feature in videoFeatures:
                trainingData.append(feature)
                trainingLabels.append(genreIndex)
    trainingData = np.array(trainingData)
    trainingLabels = np.array(trainingLabels)
    print trainingData.shape
    print trainingLabels.shape

    print "Training..."
    model = RF(n_estimators=15, n_jobs=-1).fit(trainingData, trainingLabels)
    dump_pkl(model, "RF_histogram")


def ultimate_evaluate(model):
    genres = ['action','drama','horror','romance']
    testingData = []
    testingLabels = []
    total = defaultdict.fromkeys(range(len(genres)),0)
    correct = defaultdict.fromkeys(range(len(genres)),0)
    yTrue, yPredict = [], []
    for genreIndex, genre in enumerate(genres):
        try:
            genreFeatures = load_pkl(genre+"_histogram_test")
            genreFeatures = np.array([np.array(f) for f in genreFeatures]) # numpy hack
        except Exception as e:
            print e
            return
        print "OK."
        for videoFeatures in genreFeatures:
            total[genreIndex]+=1
            d = defaultdict(int)
            predictedClasses = model.predict(videoFeatures) #List of predictions, per-frame
            print predictedClasses
            for i in predictedClasses:
                d[i]+=1
            predictedGenre = max(d.iteritems(), key=lambda x: x[1])[0]
            yPredict.append(predictedGenre)
            yTrue.append(genreIndex)
            if predictedGenre == genreIndex:
                correct[genreIndex]+=1

    print correct, total

    confusionMatrix = confusion_matrix(yTrue, yPredict)
    print confusionMatrix


if __name__=="__main__":
    """Baseline function for evaluation"""
    genres = ['action','drama','horror','romance']
    for genre in genres:
        gather_histogram_data(genre)
    train_model(genres)
    model = load_pkl('RF_histogram')
    ultimate_evaluate(model)
