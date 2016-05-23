import numpy as np
import cv2
from matplotlib import pyplot as plt


# create capture from webcam
cap = cv2.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    b,g,r = cv2.split(frame)
    
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # get gradient for x axis and y axis for each color channel using sobel operator
    sobelrx = cv2.Sobel(r,cv2.CV_64F,1,0,ksize=3)
    sobelry = cv2.Sobel(r,cv2.CV_64F,0,1,ksize=3)
    sobelgx = cv2.Sobel(g,cv2.CV_64F,1,0,ksize=3)
    sobelgy = cv2.Sobel(g,cv2.CV_64F,0,1,ksize=3)
    sobelbx = cv2.Sobel(b,cv2.CV_64F,1,0,ksize=3)
    sobelby = cv2.Sobel(b,cv2.CV_64F,0,1,ksize=3)

    # take maximum of the 3 channels
    sobelx = np.maximum(sobelrx,sobelbx,sobelgx)
    sobely = np.maximum(sobelry,sobelby,sobelgy)

    # taking orientation and converting it to degrees
    sobel = 360*(np.arctan2(sobely,sobelx))/(2*np.pi)
    sobel = sobel+180
    x=0
    
    # quantizing the orientation into 8 direction
    sobel = 8*sobel/360
    sobel = sobel.astype(int)

    # converting to bitstrings to spread the orientation
    sobel[sobel==0] = 1
    sobel[sobel==1] = 2
    sobel[sobel==2] = 4
    sobel[sobel==3] = 8
    sobel[sobel==4] = 16
    sobel[sobel==5] = 32
    sobel[sobel==6] = 64
    sobel[sobel==7] = 128
    sobel[sobel==8] = 256

    # showing sobelx and sobely
    cv2.imshow('frame1',sobelx)
    cv2.imshow('frame2',sobely)

    cv2.imshow('original',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # print x
    # z=0;    
cap.release()
cv2.destroyAllWindows()