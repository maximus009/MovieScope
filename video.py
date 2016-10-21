import cv2
import numpy as np
import sys,os
from glob import glob
from config.resources import video_resource

def get_frames(videoPath, time_step=2000):

    try:
        cap = cv2.VideoCapture(videoPath)
        for k in range(5000, 120001, time_step):
            cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, k)
            success, frame = cap.read()
            frame = cv2.resize(frame,(224,224))
            yield frame
    except Exception as e:
        print e
        return


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


get_videos(sys.argv[1])
