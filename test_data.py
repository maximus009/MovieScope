"""This is where we test data"""
import os
from video import get_frames
from model_utils import get_features_batch
from config.global_parameters import default_model_name
from config.resources import video_resource
from glob import glob
from utils import dump_pkl

def gather_training_data(genre, model_name=default_model_name):
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

if __name__=="__main__":
    genres = ['action','drama','fantasy','horror','romance']
    for genre in genres:
        gather_training_data(genre)
