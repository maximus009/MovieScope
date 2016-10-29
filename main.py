from glob import glob
from cPickle import load, dump
import numpy as np

from config.global_parameters import frameWidth, frameHeight
from config.resources import video_resource

from video import extract_feature_video 

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



if __name__=="__main__":
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
