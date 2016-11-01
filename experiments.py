from keras.models import load_model
from sys import argv
from glob import glob
import numpy as np

from video import extract_feature_video, gather_videos

def load_moviescope_model(modelName, verbose=True):

    if verbose:
        print "Loading model:",modelName
    model = load_model(modelName+'.h5')
    return model


def trial_video(videoFeatures):

    predictions = model.predict_classes(videoFeatures, verbose=0)
    prediction_scores = model.predict(videoFeatures,verbose=0)
    prediction_scores = np.mean(prediction_scores, axis=0)
    predCount = np.bincount(predictions)
    return prediction_scores, predCount
            

def trials(genre):
    genreFeatures = gather_videos(genre,limit_videos=40)
    for videoFeatures in genreFeatures:
        count = trial_video(videoFeatures)
        print count

    

if __name__=="__main__":
    model = load_moviescope_model('all_bs_16_ep_50_nf_35')
    print trial_video(np.array(list(extract_feature_video(argv[1]))))
