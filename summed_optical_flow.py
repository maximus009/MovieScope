import cv2
import numpy as np
import sys,os
from glob import glob
from config.resources import video_resource
from config.global_parameters import frameWidth, frameHeight
import scenedetect
from utils import dump_pkl, load_pkl
from model_utils import optical_flow_model
from video import sequencify


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
        for i in range(start, end+1, 30):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES,i)
            success,frame = video_capture.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            frame = cv2.resize(frame, (frameWidth, frameHeight))
            scene.append(frame)
        yield scene

def stacked_scene_optical_flow(scene):

    stackedOpticalFlow = np.zeros((2, frameWidth, frameHeight))
    for i in range(1,len(scene)):
        img = scene[i]
        prev = scene[i-1]
        flow = cv2.calcOpticalFlowFarneback(prev, img, 0.5, 3, 15, 3, 5, 1.2, 0)
        del img
        del prev
        flow = np.reshape(flow ,(2, frameWidth, frameHeight))
        stackedOpticalFlow += flow
        del flow
    return stackedOpticalFlow


def optical_flow(videoPath):

    shot_boundaries = scene_detect(videoPath)
    opticalFlow = []
    for scene in extract_scenes(videoPath, shot_boundaries):
        opticalFlow.append(stacked_scene_optical_flow(scene))
        
    opticalFlow = np.array(opticalFlow)
    print opticalFlow.shape
    return opticalFlow


def gather_optical_flow_features(genre, limit_videos = None):
    genre_OF_features = []
    videoPaths = glob(os.path.join(video_resource,'train',genre)+'/*')[:limit_videos]
    for videoPath in videoPaths:
        videoFeatures = optical_flow(videoPath)
        print videoFeatures.shape
        genre_OF_features.append(videoFeatures)
        print "*"*90
    dump_pkl(genre_OF_features, genre+"_ultimate_OF")


def create_model(genres):

    trainingData = []
    trainingLabels = []

    number_of_classes=len(genres)

    for genreIndex, genre in enumerate(genres):
        try:
            genreFeatures = load_pkl(genre+"_ultimate_OF")
        except Exception as e:
            print e
            return
        for videoFeatures in genreFeatures:
            print videoFeatures.shape
            if videoFeatures.shape==(0,):
                continue
            for scene in videoFeatures:
                for sequence in sequencify(scene,4):
                    trainingData.append(sequence)
                    trainingLabels.append(genreIndex)

    trainingData = np.array(trainingData)
    trainingLabels = np.array(trainingLabels)
    print trainingData.shape
    print trainingLabels.shape
    model = optical_flow_model(number_of_classes)
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(trainingData, trainingLabels)

    

if __name__=="__main__":

    create_model(['action','drama','horror','romance'])
