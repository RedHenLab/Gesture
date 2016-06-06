import numpy as np
from matplotlib import pyplot as plt
import cv2
import sys

# reading face and eye cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# making video capture using filename or webcam
if (len(sys.argv)>1): cap = cv2.VideoCapture(sys.argv[1])
else: cap= cv2.VideoCapture(0)
thresh  = None
roihist = None
# starting the while loop
while(True):

    # taking capture
    ret,img = cap.read()
    
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #detecting face and making rectangle around it
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        val, mask = cv2.threshold(roi_gray, 145, 255, cv2.THRESH_BINARY)
        
        roi_hsv = cv2.cvtColor(roi_color,cv2.COLOR_BGR2HSV)

        #getting histogram of face for backprojection
        roihist = cv2.calcHist([roi_hsv],[0, 1], mask, [180, 256], [0, 180, 0, 256] )

        cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    
    if roihist is not None:
        # back projecting face histogram
        dst = cv2.calcBackProject([hsv],[0,1],roihist,[0,180,0,256],1)
    
        # running an ellipse structuring element for filtering 
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        cv2.filter2D(dst,-1,disc,dst)
    
        # taking threshold to get binary 
        ret,thresh = cv2.threshold(dst,50,255,0)
        
        kernel = np.ones((4,4),np.uint8)
        #fgmask = cv2.erode(fgmask,kernel,iterations= 1)
        #fgmask = cv2.dilate(fgmask,kernel,iterations= 2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.GaussianBlur(thresh,(5,5),0)
    
        # finding contours
        im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        # Draw the contour on the source image
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > 1700:
                ellipse = cv2.fitEllipse(c)
                cv2.ellipse(img,ellipse,(0,255,0),2)
    
        thresh = cv2.merge((thresh,thresh,thresh))
                #res = cv2.bitwise_and(img,thresh)
            # res = np.vstack((img,thresh,res))
    
    
            # chans = cv2.split(roi_color)
            # colors = ("b", "g", "r")
            # for (chan, color) in zip(chans, colors):
            #     # create a histogram for the current channel and
            #     # concatenate the resulting histograms for each
            #     # channel
            #     hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            #     features.extend(hist)
             
            #     # plot the histogram
            #     plt.plot(hist, color = color)
            #     plt.xlim([0, 256])
    
        # showing various thresh and current frame
        cv2.imshow('result',thresh)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
