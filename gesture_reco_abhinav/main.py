#gesture tracking using skin colour detection
import sys
import cv2
import numpy as np
from numpy import sqrt,arccos,rad2deg

#Global Constants
SIZE_OF_DATA = 25   # number of data points
M = 25              # number of rows in each datapoint (basically time series latency)
N = 8               # number of features captures at each time instance
S = 5               # number of eigen vectors selected from row-row covariance matrix
R = 5               # number of eigen vectors selected from column-column covariance matrix. SXR will be the size of the each datapoint after 2D SVD (dimensionality reduction).



################################################################################### Function Definitions ####################################################################################################

# function to get euclidean distance
def Euclidean(a,b,c,d):
    return ((a-b)**2 + (c-d)**2)**0.5



# function to get distance for knn method
def distance(a1,a2):
    result=0.0
    a1 = np.array(a1)
    a2 = np.array(a2)
    for x in range(0,R):
        temp=0
        for y in range(0,S):
            d = a1[y][x]-a2[y][x]
            temp=temp + (d)**2

        result = result + temp**0.5

    return result 

# Train function: To get the row-row and column-column covariance matrix after training. Takes as input training data. Returns U(MXS) row-row covariance matrix, V(RXN) column-column 
# covariance matrix and M(S*R) new mapped data.
def train(data):
    
    avg = [[0 for x in range(N)] for y in range(M)]

    # get averag of whole training data
    for x in range(0,SIZE_OF_DATA):
        for y in range(0,M):
            for z in range(0,N):
                avg[y][z] = avg[y][z] + data[x][y][z]/SIZE_OF_DATA

    avg = np.mat(avg)
    data = np.array(data)
    f = [[0 for x in range(M)] for y in range(M)]                
    g = [[0 for x in range(N)] for y in range(N)]
    f = np.mat(f)
    g = np.mat(g)
    # f=row-row and g = column column covariance matrices
    for x in range(SIZE_OF_DATA):
        temp = data[x]
        temp = np.mat(temp)
        f = f + (np.mat(temp-avg))*(np.transpose(np.mat(temp-avg)))
        g = g + (np.transpose(np.mat(temp-avg)))*(np.mat(temp-avg))
    f = f/SIZE_OF_DATA
    g = g/SIZE_OF_DATA
    # get eigenVec and Eigen Values
    feignV,feignVec = np.linalg.eig(f)
    geignV,geignVec = np.linalg.eig(g)

    # choose top 5 and 3 most prominent eigen vectors
    U = feignVec[0:M,0:S]
    V = geignVec[0:N,0:R]
    U = np.transpose(U)
    M = [[[0 for x in range(R)] for y in range(S)] for z in range(SIZE_OF_DATA)]
    M = np.array(M)

    # map training data to new dimensions
    for x in range(SIZE_OF_DATA):
        temp = data[x]
        temp = np.mat(temp)
        temp = (U*temp)*V
        M[x] = np.array(temp)

    return M,U,V

# Read Function: To read the training data. Take an empty 2d list and name of the indexfile (file with names of files for each data point) as input.
def read(data,indexfilename):
    indexfile = open(indexfilename,"r")
    count=0
    for filename in indexfile:
        filename = filename.rstrip()
        # print filename
        f = open(filename,"r")
        for x in f:
            x=x.rstrip()
            x=x.split()
            data[count].append([int(x[0]),int(x[1]),int(x[2]),int(x[3]),int(x[4]),int(x[5]),float(x[6]),float(x[7])])

        count=count+1
        f.close()
    return data


# Classify Function : Classifies given data using 2D svd and k-nearest neighbours. Newdata is the new datapoint (to be classified) of size m*n.
def classify(newData):
    data = [[] for x in range(SIZE_OF_DATA)]
    # read training data
    data = read(data,"ulti.txt")
    # apply 2d SVD
    M,U,V = train(data)
    # print M[1].shape
    temp = np.mat(newData)
    temp = (U*temp)*V
    ind=0
    mini=1500000000000
    distlist = []
    indlist = []

    # apply knn
    for x in range(SIZE_OF_DATA):
        dist = distance(M[x],temp)
        # print dist
        if(len(distlist)<3):
            distlist.append(dist)
            indlist.append(x)
        else:
            i = distlist.index(max(distlist))
            if(distlist[i]>dist):
                distlist[i]=dist
                indlist[i]=x
        
        if(dist<mini):
            mini=dist
            ind=x

    for x in range(len(indlist)):
        indlist[x] = indlist[x]%5

    i = max(set(indlist),key=indlist.count)

    print i

    #classify
    if(ind<5 ): print "Foldedhands"
    elif(ind>=5 and ind<10 ): print "Handface"
    elif(ind>=10 and ind<15): print "Pointing"
    elif(ind<20): print "Beginning to end" 
    else: print "Clapping" 
    # print ind,mini


# getSkin fuction: To filter skin from the given image using HSV range. Input is a rgb image. Output is a filtered binary image with skin area marked as white.  
def getSkin(image):

    """OpenCV uses different ranges for HSV as compared to other applications. h:0-130 s:0-255 v:0-255 
    hsv color range for use in OpenCV [0,30,60 - 20,150,255] OR [0,40,60-20,150,255] OR [0,10,60-20,150,255]
    NOTE: since skin color can have a wide range, can use markers on finger tips to target a smaller and easy to use color range """
    MIN = np.array([0,40,60],np.uint8)
    MAX = np.array([20,150,255],np.uint8) #HSV: V-79%
    HSVImg = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    
    # Filtering the image for given HSV range and then  applying Morphological Operations
    filterImg = cv2.inRange(HSVImg,MIN,MAX) #filtering by skin color
    filterImg = cv2.erode(filterImg,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))) #eroding the image
    filterImg = cv2.dilate(filterImg,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))) #dilating the image

    return filterImg

#getFaceCoordinate function: To find the center coordinates of the face using face_cascade. Inputs are a grayscale image and face_cascade. Outputs are the center coordinates
def getFaceCoodinates(gray,face_cascade):
    fx=0
    fy=0
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        fx = x+w/2
        fy= y+h/2

    return (fx,fy)

# divideImg function: To divide the image on the basis of y coordinate of the face. Inputs include a binary image and the face coordinates. Returns 2 new images one for left and one for right side of the face. 
def divideImg(image,fx,fy):
    if(fy==0):
        thresh1 = image[0:x,0:y/2]
        thresh2 = image[0:x,y/2:y] 
    else :
        thresh1 = image[0:x,0:fy]
        thresh2 = image[0:x,fy:y] 

    return (thresh1,thresh2)


# getHand function: To get the hand features (centers coordinates, area, ellipse) from the thresholded image. Takes as input the threshold image. Returns the hand features along 
# with a flag for valid hand detection
def getHand(thresh):
    a,contours, heirarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    max_area=-1
    max_contour = None
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > max_area:
            # fit and draw ellipse
            max_contour = c
            max_area = area

    if(max_area>1000):
        ellipse1 = cv2.fitEllipse(max_contour1)
        # cv2.ellipse(OriginalImg,ellipse1,(0,255,0),2)
        M = cv2.moments(max_contour1)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        # mssg = str(cx) + " " + str(cy)+"\n"
        # data.write(mssg)

    else: 
        flag=-1
        cx=10**9
        cy=10**9
        ellipse = [10**9,10**9,10**9]


    return (cx,cy,max_area,ellipse,flag)



################################################################################# Main Loop ##################################################################################################

# Starting the camera/video capture
if (len(sys.argv)>1): camera = cv2.VideoCapture(sys.argv[1])
else: 
camera= cv2.VideoCapture(0)

#if(len(sys.argv)>1):
#    data1 = open(sys.argv[1],"w")
data = []

# making all the windows
# cv2.namedWindow("skinimage",cv2.WINDOW_NORMAL)
cv2.namedWindow("original",cv2.WINDOW_NORMAL)
# cv2.namedWindow("Segmentation",cv2.WINDOW_NORMAL)
cv2.namedWindow("Foreground",cv2.WINDOW_NORMAL)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initiating the Background Subtractor

fgbg = cv2.createBackgroundSubtractorMOG2(500,16,0)

count=0
# While Loop
while count<M:
    #run()
    # Reading from capture
    flag1=flag2=0
    ret, image = camera.read()

    if image is None:
        print "Video Over"
        break
    # Applying background Subtraction
    fgmask = fgbg.apply(image.copy())    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    OriginalImg = image.copy()
    image = cv2.blur(image,(5,5)) #try blurring with different kernel sizes to check effectiveness
    # skin detection using global HSV range
    skinImg = getSkin(image)
    # get face center coordinates
    (fx,fy) = getFaceCoodinates(gray,face_cascade)
    # Taking and of skin image and foreground image
    thresh = cv2.bitwise_and(skinImg,fgmask)
    thresh = cv2.dilate(thresh,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))) #dilating the image
    ret,thresh = cv2.threshold(thresh,50,255,cv2.THRESH_BINARY)
    (x,y) = np.array(thresh).shape
    
    (thresh1,thresh2) = divideImg(thresh,fx,fy)
    
    #getting hand contour features
    (cx1,cy1,area1,ellipse1,flag1) = getHand(thresh1)
    (cx2,cy2,area2,ellipse2,flag2) = getHand(thresh2)

    if((flag1!=-1 or flag2 !=-1) and (fx!=0 and fy!=0)):
        # mssg = str(cx1) + " " + str(cy1)+" "+str(fx) + " " + str(fy)+" "+str(cx2) + " " + str(cy2)+" "+str(ellipse1[2])+" "+str(ellipse2[2])+"\n"
        # data1.write(mssg)
        mssg = [cx1,cy1,fx,fy,cx2,cy2,ellipse1[2],ellipse2[2]]
        data.append(mssg)
        count=count+1
    
    if(count==M):
        classify(np.array(data))
        data = []
        count=0
        # break

    
    # cv2.drawContours(OriginalImg,contours,-1,(255,0,0),-2) #blue color - marks hands

    # Show all the images
    # cv2.imshow("Segmentation",thresh)
    cv2.imshow("Foreground",thresh)
    cv2.imshow("original",OriginalImg)
    if cv2.waitKey(10)==27:
        break
cv2.destroyAllWindows()
# data1.close()
