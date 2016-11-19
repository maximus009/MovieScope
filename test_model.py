"""This is where we test data"""
import os
from video import sequencify
from config.global_parameters import default_model_name
from glob import glob
import numpy as np
from utils import dump_pkl, load_pkl, load_moviescope_model
from collections import defaultdict
from sklearn.metrics import confusion_matrix


def ultimate_evaluate(model):
    genres = ['action','horror','romance']
    testingData = []
    testingLabels = []
    total = defaultdict.fromkeys(range(len(genres)),0)
    correct = defaultdict.fromkeys(range(len(genres)),0)
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
            sequences = np.array(list(sequencify(videoFeatures)))
            num_of_samples = len(sequences)
            num_of_frames = len(sequences[0])
            testingDataTensor = np.zeros((num_of_samples, num_of_frames, 4096))
            print num_of_samples
            for sampleIndex in range(num_of_samples):
                for vectorIndex in range(num_of_frames):
                    try:
                        testingDataTensor[sampleIndex][vectorIndex] = sequences[sampleIndex][vectorIndex]
                    except Exception as e:
                        continue

            try:
                predictedClasses = model.predict_classes(testingDataTensor) #List of predictions, per-frame
                print predictedClasses
                for i in predictedClasses:
                    d[i]+=1
                predictedGenre = max(d.iteritems(), key=lambda x: x[1])[0]
                yPredict.append(predictedGenre)
                yTrue.append(genreIndex)
                if predictedGenre == genreIndex:
                    correct[genreIndex]+=1
            except Exception as e:
                print e

    print correct, total

    confusionMatrix = confusion_matrix(yTrue, yPredict)
    print confusionMatrix

#
if __name__=="__main__":

    from sys import argv
    model = load_moviescope_model(argv[1])
    ultimate_evaluate(model)
