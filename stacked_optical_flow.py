import cv2
import numpy as np
import sys,os
from glob import glob
from config.resources import video_resource
from config.global_parameters import frameWidth, frameHeight
import scenedetect
from utils import dump_pkl


def scene_detect(videoPath):
    sceneDetect = []
    detector_list = [scenedetect.detectors.ThresholdDetector(threshold = 30, min_percent = 0.9)]
    print videoPath
    video_framerate, frames_read = scenedetect.detect_scenes_file(videoPath, sceneDetect, detector_list)
    return sceneDetect

def extract_scenes(videoPath, shot_boundaries):
    video_capture = cv2.VideoCapture(videoPath)

    for i in range(len(shot_boundaries)-1):
        start = shot_boundaries[i]
        end = shot_boundaries[i+1]
        scene = []
        video_capture.set(cv2.CAP_PROP_POS_FRAMES,start)
        for i in range(start, end):
            success,frame = video_capture.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            frame = cv2.resize(frame, (frameWidth, frameHeight))
            scene.append(frame)
        yield scene

def stacked_scene_optical_flow(scene):

    stackedOpticalFlow = []
    for i in range(1,len(scene)):
        img = scene[i]
        prev = scene[i-1]
        flow = cv2.calcOpticalFlowFarneback(prev, img, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = np.reshape(flow ,(2, frameWidth, frameHeight))
        flowX, flowY = flow[0],flow[1]
        stackedOpticalFlow.append(flowX)
        stackedOpticalFlow.append(flowY)
        del flow
    stackedOpticalFlow = np.array(stackedOpticalFlow)
    return stackedOpticalFlow


def optical_flow(videoPath):

    shot_boundaries = scene_detect(videoPath)
    opticalFlow = []
    for scene in extract_scenes(videoPath, shot_boundaries):
        opticalFlow.append(stacked_scene_optical_flow(scene))
        
    opticalFlow = np.array(opticalFlow)
    return opticalFlow


def gather_optical_flow_features(genre):
    genre_OF_features = []
    videoPaths = glob(os.path.join(video_resource,'train',genre)+'/*')
    for videoPath in videoPaths:
        genre_OF_features.append(optical_flow(videoPath))
    dump_pkl(genre_OF_features, genre+"_ultimate_OF")


if __name__=="__main__":
    genres = ['action']
    for genre in genres:
        gather_optical_flow_features(genre)


