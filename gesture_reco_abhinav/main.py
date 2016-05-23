import numpy as np
import cv2
from matplotlib import pyplot as plt


cap = cv2.VideoCapture(0)
# z=1;
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    b,g,r = cv2.split(frame)
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    
    sobelrx = cv2.Sobel(r,cv2.CV_64F,1,0,ksize=3)
    sobelry = cv2.Sobel(r,cv2.CV_64F,0,1,ksize=3)
    sobelgx = cv2.Sobel(g,cv2.CV_64F,1,0,ksize=3)
    sobelgy = cv2.Sobel(g,cv2.CV_64F,0,1,ksize=3)
    sobelbx = cv2.Sobel(b,cv2.CV_64F,1,0,ksize=3)
    sobelby = cv2.Sobel(b,cv2.CV_64F,0,1,ksize=3)

    sobelx = np.maximum(sobelrx,sobelbx,sobelgx)
    sobely = np.maximum(sobelry,sobelby,sobelgy)

    sobel = 360*(np.arctan2(sobely,sobelx))/(2*np.pi)
    sobel = sobel+180
    x=0
    sobel = 8*sobel/360
    # cv2.blur(sobel,(3,3))
    sobel = sobel.astype(int)

    sobel[sobel==0] = 1
    sobel[sobel==1] = 2
    sobel[sobel==2] = 4
    sobel[sobel==3] = 8
    sobel[sobel==4] = 16
    sobel[sobel==5] = 32
    sobel[sobel==6] = 64
    sobel[sobel==7] = 128
    sobel[sobel==8] = 256
    # sobel = sobel.astype(str)
    sobel = cv2.blur(sobel,(3,3))
    sobel = sobel*9
    print sobel[1]
    sobel = sobel.astype(float)

    cv2.imshow('frame1',sobel)
    
    sobel = sobel.astype(int)
    
    cv2.imshow('frame2',sobely)
    cv2.imshow('original',r)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # print x
    # z=0;    
cap.release()
cv2.destroyAllWindows()