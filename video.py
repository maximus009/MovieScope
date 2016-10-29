import cv2
import numpy as np
import sys,os
from glob import glob
from config.resources import video_resource
from config.global_parameters import frameWidth, frameHeight
from model_utils import get_features

def get_frames(videoPath, start_time=5000, end_time=120000, time_step=2000):

    try:
        cap = cv2.VideoCapture(videoPath)
        for k in range(start_time, end_time+1, time_step):
            cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, k)
            success, frame = cap.read()
            if success:
                frame = cv2.resize(frame,(frameWidth, frameHeight))
                yield frame
    except Exception as e:
        print e
        return


def extract_feature_video(videoPath, verbose=False):

    """Returns features of shape (N, 4096), N: number of processed frames"""
    if verbose:
        print "Starting to extract features for ",videoPath
    for frame in get_frames(videoPath):
        feature = get_features(frame)
        if verbose:
            print ".",
        yield feature


def save_frames_video(videoPath, videoID, outPath='./data'):

    if not os.path.exists(outPath):
        os.makedirs(outPath)
    
    if not os.path.exists(videoPath):
        print "Video not found"
        return None

    for frame_no, frame in enumerate(get_frames(videoPath)):
        frameWritePath = os.path.join(outPath,str(videoID)+'_'+str(frame_no)+'.jpg')
        print 'writing frame# {0} to {1}'.format(frame_no, frameWritePath)
        cv2.imwrite(frameWritePath, frame)


def get_videos(genre):
    
    videoPaths = glob(video_resource+genre+'/*')
    for videoID, videoPath in enumerate(videoPaths):
        print videoPath
        save_frames_video(videoPath, videoID, outPath='./data/'+genre)


if __name__=="__main__":
    get_videos(sys.argv[1])
