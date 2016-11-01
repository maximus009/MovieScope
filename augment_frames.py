import cv2, sys, os
from config.global_parameters import frameWidth, frameHeight
from utils import sliding_window

def augment(frame, flip=True, patches=True):

    """Given an image, performs relevant augmentations to increase training data"""
    if not patches:
        #frame considered as a single input, no data augmentation
        yield frame
    else:
        #patchify the entire frame to produce crops of the frame 
        cropWidth, cropHeight = int(0.8*frameWidth), int(0.8*frameHeight)
        for x,y,croppedImage in sliding_window(frame, windowSize=(cropWidth, cropHeight), horizontalStride=16, verticalStride=32):
            if croppedImage.shape != (cropHeight,cropWidth,3):
                continue
            resizedCroppedImage = cv2.resize(croppedImage, (frameWidth, frameHeight), interpolation=cv2.INTER_CUBIC)
            if not flip:
                yield resizedCroppedImage, None 
            else:
                yield resizedCroppedImage, cv2.flip(resizedCroppedImage, 1)
                 

def augment_and_save(baseDir = 'results', videoDir = '.'):

    ctr = 0
    imagePath = sys.argv[1]
    image = cv2.imread(imagePath)
    imageName = imagePath.split('/')[-1].split('.')[0]
    outPath = os.path.join(baseDir, videoDir, imageName)
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    for image, flippedImage in augment(image, flip=True, patches=True):
        cv2.imwrite(outPath+"/augTest_"+str(ctr)+".jpg", image)
        if flippedImage is not None: 
            cv2.imwrite(outPath+"/results/augTest_flip_"+str(ctr)+".jpg", flippedImage)
        ctr+=1


if __name__=="__main__":

    augment_and_save(videoDir='myFairLady')
