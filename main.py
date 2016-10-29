from glob import glob
import numpy as np

from utils import load_pkl, dump_pkl
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
    for r in romanceFeatures:
        print len(r),
    print
    for h in horrorFeatures:
        print len(h)


if __name__=="__main__":
    train()

