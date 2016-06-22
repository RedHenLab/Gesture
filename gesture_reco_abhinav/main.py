#gesture tracking using skin colour detection
import sys
import cv2
import numpy as np
from numpy import sqrt,arccos,rad2deg

# Starting the camera/video capture
if (len(sys.argv)>1): camera = cv2.VideoCapture(sys.argv[1])
else: camera= cv2.VideoCapture(0)

# making all the windows
cv2.namedWindow("skinimage",cv2.WINDOW_NORMAL)
cv2.namedWindow("original",cv2.WINDOW_NORMAL)
cv2.namedWindow("Segmentation",cv2.WINDOW_NORMAL)
cv2.namedWindow("Foreground",cv2.WINDOW_NORMAL)

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initiating the Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# While Loop
while True:
    #run()
    # Reading from capture
    ret, image = camera.read()

    if image is None:
        print "Video Over"
        break
    # Applying background Subtraction
    fgmask = fgbg.apply(image.copy(),1)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    OriginalImg = image.copy()
    NoFilterImg = image.copy()

    image = cv2.blur(image,(5,5)) #try blurring with different kernel sizes to check effectiveness

    # skin detection using global HSV range
    """OpenCV uses different ranges for HSV as compared to other applications. h:0-180 s:0-255 v:0-255 
    hsv color range for use in OpenCV [0,30,60 - 20,150,255] OR [0,40,60-20,150,255] OR [0,10,60-20,150,255]
    NOTE: since skin color can have a wide range, can use markers on finger tips to target a smaller and easy to use color range """
    MIN = np.array([0,40,60],np.uint8)
    MAX = np.array([20,150,255],np.uint8) #HSV: V-79%
    HSVImg = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # cv2.imshow("img3",HSVImg)
    
    # Filtering the image for given HSV range and then  applying Morphological Operations
    filterImg = cv2.inRange(HSVImg,MIN,MAX) #filtering by skin color

    filterImg = cv2.erode(filterImg,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))) #eroding the image

    filterImg = cv2.dilate(filterImg,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))) #dilating the image


    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Show filteres skin image
    cv2.imshow("skinimage",filterImg)

    
    # Taking and of skin image and foreground image
    thresh = cv2.bitwise_and(filterImg,fgmask)

    thresh = cv2.erode(thresh,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))) #eroding the image

    thresh = cv2.dilate(thresh,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))) #dilating the image

    ret,thresh = cv2.threshold(thresh,50,255,cv2.THRESH_BINARY)

    #getting all the contours
    a,contours, heirarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 2000:
            # fit and draw ellipse
            ellipse = cv2.fitEllipse(c)
            cv2.ellipse(OriginalImg,ellipse,(0,255,0),2)
            # print ellipse[1]
            # break

    # cv2.drawContours(OriginalImg,contours,-1,(255,0,0),-2) #blue color - marks hands

    # Show all the images
    cv2.imshow("Segmentation",thresh)
    cv2.imshow("Foreground",fgmask)
    cv2.imshow("original",OriginalImg)
    if cv2.waitKey(10)==27:
        break