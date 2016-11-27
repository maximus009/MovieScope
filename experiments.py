from keras.models import load_model
from sys import argv
from glob import glob
import numpy as np

from utils import load_moviescope_model

from video import extract_feature_video, gather_videos


def trial_video(videoFeatures):

    predictions = model.predict_classes(videoFeatures, verbose=0)
    print "predictions:",predictions,
    prediction_scores = model.predict(videoFeatures,verbose=0)
    prediction_scores = np.mean(prediction_scores, axis=0)
    predCount = np.bincount(predictions)
    return prediction_scores, predCount
            
def test_video(videoPath):
    videoFeatures = np.array(list(extract_feature_video(videoPath, verbose=True)))
    print trial_video(videoFeatures)


def test_videos(testPath):
    videoPaths = glob(testPath+'/*')
    for videoPath in videoPaths:
        test_video(videoPath)


def trials(genre):
    genreFeatures = gather_videos(genre,limit_videos=40)
    for videoFeatures in genreFeatures:
        count = trial_video(videoFeatures)
        print count


if __name__=="__main__":
    model = load_moviescope_model('spatial_3g_bs32_ep50_nf_75')
#    model = load_moviescope_model('all_bs_16_ep_50_nf_35')
    from time import time
    s = time()
    test_video(argv[1])
    print time()-s,"seconds."
