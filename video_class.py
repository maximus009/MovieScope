import cv2
import numpy as np
import sys,os
from glob import glob
from config.resources import video_resource
from config.global_parameters import frameWidth, frameHeight


class genre:
    genre=""
    path=""
    no_videos=0

    def __init__(self,genre,limit_videos=-1):
        self.genre=genre
        self.no_videos=limit_videos
        self.path=video_resource
    def gather_videos(self):
        videoPaths = glob(self.path+'\\'+self.genre+'/*')
        videoPaths=videoPaths[:self.no_videos]
        return videoPaths

class video:
    
    def __init__(self,path):
        self.path=path
        self.frames=[]

    def get_frames(self, start_time=5000, end_time=120000, time_step=2000):
        try:
            #print videoPath
            cap = cv2.VideoCapture(self.path)
            for k in range(start_time, end_time+1, time_step):
                cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, k)
                success, frame = cap.read()
                if success:
                    frame = cv2.resize(frame,(frameWidth, frameHeight))
                    self.frames.append(frame)
            return self.frames
        except Exception as e:
            print e
            return
        
    def get_features(self):
        videoFeatures=self.extract_feature_video(verbose=True)
        return videoFeatures


    def extract_feature_video(self,verbose=False):
        """Returns features of shape (N, 4096), N: number of processed frames"""
        if verbose:
            print "\nStarting to extract features for ",self.path
        for frame in self.frames:
            #feature = get_features(frame)
            if verbose:
                print ".",
        return "Return Feature here"
        print
    

if __name__=="__main__":
    horror=genre('horror',limit_videos=10)
    video_list=horror.gather_videos()
    for each_video in video_list:
        obj=video(each_video)
        frames_each_video=obj.get_frames()
        #features_each_video=obj.get_features()
        print
        print
        print "Here lies frames for each video",len(frames_each_video)
        print
        print "%"*50
       # print "Here lies features for each video",features_each_video