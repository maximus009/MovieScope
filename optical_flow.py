from glob import glob
import cv2
from config.resources import video_resource
from sys import argv
import numpy as np


def draw_flow(img, flow, step=16):
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


def optical_flow(video, verbose = False, visualize=False):


    if verbose:
        print "Stacking Optical Flow features for",video

    stacked_optical_flow = []
    cap = cv2.VideoCapture(video)
    ret, prev = cap.read()
    prev = cv2.resize(prev, (224,224)) 
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while(True):
        if verbose:
            print ".",
        ret,img = cap.read()
        if ret is not True:
            if verbose:
                print "Frames processed."
            break
        img = cv2.resize(img, (224, 224))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
        stacked_optical_flow.append(flow)
        prevgray = gray
        if visualize:
            cv2.imshow('flow', draw_flow(gray, flow))
            if cv2.waitKey(10) == ord('q'):
                break
    print
    cv2.destroyAllWindows()
    stacked_optical_flow = np.array(stacked_optical_flow)
    return stacked_optical_flow


def main(genre):
    videoPaths = glob(video_resource+genre+'/*')
    for videoPath in videoPaths:
        video_OF_feature = optical_flow(videoPath, verbose=True, visualize=True)
        print video_OF_feature.shape
        break


if __name__ == '__main__':
    main('romance')
