import cv2
import numpy as np
import sys,os
from glob import glob
from config.resources import video_resource
from config.global_parameters import frameWidth, frameHeight
import scenedetect


class genre:
    genre=""
    path=""
    no_videos=0

    def __init__(self,genre,limit_videos=-1):
        self.genre=genre
        self.no_videos=limit_videos
        self.path=video_resource
    def gather_videos(self):
        videoPaths = glob(self.path+'\\'+self.genre+'\*')
        videoPaths=videoPaths[:self.no_videos]
        return videoPaths

class video:
    
    def __init__(self,path):
        self.path=path
        self.frames=[]
        self.frames_spatial=[]

    def get_frames(self, start_time=5000, end_time=120000, time_step=2000):
        try:
            #print videoPath
            cap = cv2.VideoCapture(self.path)
            for k in range(start_time, end_time+1, time_step):
                cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, k)
                success, frame = cap.read()
                if success:
                    frame = cv2.resize(frame,(frameWidth, frameHeight))
                    self.frames_spatial.append(frame)
            return self.frames_spatial
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

    def scene_detect(self):
        sceneDetect = []
        detector_list = [scenedetect.detectors.ThresholdDetector(threshold = 30, min_percent = 0.9)]
        print self.path
        video_framerate, frames_read = scenedetect.detect_scenes_file(self.path, sceneDetect, detector_list)
        #sceneDetect=os.system("scenedetect -i "+self.path+" -d content -t 30")
        return sceneDetect

    def stitch_video_frames(self,start,end):
        print "next set"
        scene=[]
        video_capture = cv2.VideoCapture(self.path)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES,start)
        for i in range(start, end):
            ret,frame = video_capture.read()
            #scene.append(frame)
            if not ret:
                break
        return scene


if __name__=="__main__":
    horror=genre('animation',limit_videos=10)
    video_list=horror.gather_videos()
    print video_list
    for each_video in video_list:
        obj=video(each_video)
        print obj.path
        frames_each_video=obj.get_frames()
        sceneDetect=obj.scene_detect()
        print sceneDetect
        scenes=[]
        for index in range(len(sceneDetect)-3):
            start=sceneDetect[index]
            end=sceneDetect[index+1]
            scenes.append(obj.stitch_video_frames(start,end))
        #features_each_video=obj.get_features()
        print
        print
        print "Scene detect",sceneDetect
        print
        print "%"*50
       # print "Here lies features for each video",features_each_video