"""This is where we test data"""
import os
from video import get_frames
from model_utils import get_features_batch
from config.global_parameters import default_model_name
from config.resources import video_resource
from glob import glob
import numpy as np
from utils import dump_pkl, load_pkl, load_moviescope_model
from collections import defaultdict
from sklearn.metrics import confusion_matrix


def gather_testing_data(genre, model_name=default_model_name):
    """Driver function to collect frame features for a genre"""

    testPath = os.path.join(video_resource,'test',genre)
    print testPath 
    videoPaths = glob(testPath+'/*')
    genreFeatures = []
    for videoPath in videoPaths:
        print videoPath,":",
        frames =list(get_frames(videoPath, time_step=1000))
        print len(frames),
        if len(frames)==0:
            print "corrupt."
            continue
        videoFeatures = get_features_batch(frames)
        print videoFeatures.shape
        genreFeatures.append(videoFeatures)

    outPath = genre+"_test_"+model_name
    dump_pkl(genreFeatures, outPath)

def test_video(videoPath, model):
    """Return the genre type for each video input"""
    frames = list(get_frames(videoPath, time_step=1000))
    if len(frames)==0:
        print "Error in video"
        return
    videoFeatures = get_features_batch(frames)
    predictedClasses = model.predict_classes(videoFeatures)
    predictedScores = model.predict(videoFeatures)
    return predictedClasses, predictedScores
    

def ultimate_evaluate(model):
    genres = ['action','drama','horror','romance']
    testingData = []
    testingLabels = []
    total = {0:0, 1:0, 2:0, 3:0}
    correct = {0:0, 1:0, 2:0, 3:0}
    yTrue, yPredict = [], []
    for genreIndex, genre in enumerate(genres):
#        print "Looking for pickle file: data/{0}{1}.p".format(genre, str(num_of_videos)),
        try:
            genreFeatures = load_pkl(genre+"_test_"+default_model_name)
            genreFeatures = np.array([np.array(f) for f in genreFeatures]) # numpy hack
        except Exception as e:
            print e
            return
        print "OK."
        for videoFeatures in genreFeatures:
            """to get all frames from a video -- hacky"""
            total[genreIndex]+=1
            d = defaultdict(int)
            predictedClasses = model.predict_classes(videoFeatures) #List of predictions, per-frame
            print predictedClasses
            for i in predictedClasses:
                d[i]+=1
            predictedGenre = max(d.iteritems(), key=lambda x: x[1])[0]
            yPredict.append(predictedGenre)
            yTrue.append(genreIndex)
            if predictedGenre == genreIndex:
                correct[genreIndex]+=1

    print correct, total
    yTrue = total.values()
    yPredict = correct.values()

    confusionMatrix = confusion_matrix(yTrue, yPredict)
    print confusionMatrix

#
if __name__=="__main__":

    from sys import argv
    model = load_moviescope_model(argv[1])
    ultimate_evaluate(model)
    """to call test_video"""
