import os, sys
from glob import glob
import cv2
import numpy as np
from config.global_parameters import default_model_name
from config.resources import video_resource
from model_utils import get_features_batch
from utils import dump_pkl
from video import get_frames

def gather_data():

    for genre in ['romance', 'horror']:
        genreData, genreLabels = get_features_for_genre(genre)
        dump((genreData, genreLabels), open('data/'+genre+'.p','wb'))

def gather_training_data(genre, model_name=default_model_name):
    """Driver function to collect frame features for a genre"""

    trainPath = os.path.join(video_resource,'train',genre)
    print trainPath
    videoPaths = glob(trainPath+'/*')[:5]
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

    outPath = genre+"_ultimate_"+model_name
    dump_pkl(outPath)

        
if __name__=="__main__":
    genres=['action','animation','horror','romance']
    gather_training_data(genres[0])
