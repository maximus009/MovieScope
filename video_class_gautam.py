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
        print self.path
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
            scene.append(frame)
            if not ret:
                break
        return scene

    def draw_flow(self,img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis


    def optical_flow(self,scenes, verbose = False, visualize=False):
        if verbose:
            print "Stacking Optical Flow features:"

        stacked_optical_flow = []
        prev = scenes[0]
        prev = cv2.resize(prev, (224,224)) 
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        for i in range(1,len(scenes)):
            if verbose:
                print ".",
            img = scenes[i]
            img = cv2.resize(img, (224, 224))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
            stacked_optical_flow.append(flow)
            prevgray = gray
            if visualize:
                cv2.imshow('flow', self.draw_flow(gray, flow))
                if cv2.waitKey(10) == ord('q'):
                    break
        print
        cv2.destroyAllWindows()
        stacked_optical_flow = np.array(stacked_optical_flow)
        return stacked_optical_flow


if __name__=="__main__":
    horror=genre('horror',limit_videos=1)
    video_list=horror.gather_videos()
    print video_list
    for each_video in video_list:
        optical_flow=[]
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
        for i in scenes[10:]:
            opt_flow=obj.optical_flow(i,visualize=False)
            print opt_flow
            print "Never let you go"

        #features_each_video=obj.get_features()
        print
        print
        print "Scene detect",sceneDetect
        print
        print "%"*100
       # print "Here lies features for each video",features_each_video