from utils import load_moviescope_model
from video import get_frames, sequencify
from model_utils import get_features
from sys import argv

import numpy as np

def predict_genre(videoPath):
    videoFeatures = []
    print "extracting features for",videoPath
    fixedFrames = []
    for frame in get_frames(videoPath, time_step=1000):
        frameFeatures = get_features(frame)
        videoFeatures.append(frameFeatures)
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
                print e
                continue
    print testingDataTensor.shape
    predictions = model.predict_classes(testingDataTensor)
    print predictions

if __name__=="__main__":
    from time import time
    model = load_moviescope_model("lstm_3g_bs10_ep100")
    videoPath = argv[1]
    s = time()
    predict_genre(videoPath)
    print time()-s,"seconds."
