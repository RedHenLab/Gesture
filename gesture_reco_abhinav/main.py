#gesture tracking using skin colour detection
import sys
import cv2
import numpy as np
from numpy import sqrt,arccos,rad2deg

# function to gwt euclidean distance
def Euclidean(a,b,c,d):
    return ((a-b)**2 + (c-d)**2)**0.5



# function to get distance for knn method
def distance(a1,a2,m,n):
    result=0.0
    a1 = np.array(a1)
    a2 = np.array(a2)
    for x in range(0,n):
        temp=0
        for y in range(0,m):
            d = a1[y][x]-a2[y][x]
            temp=temp + (d)**2

        result = result + temp**0.5

    return result 

# get the row-row and column column covariance matrix
def train(data,m,n,number):
    
    avg = [[0 for x in range(n)] for y in range(m)]

    # get averag of whole training data
    for x in range(0,25):
        for y in range(0,m):
            for z in range(0,n):
                avg[y][z] = avg[y][z] + data[x][y][z]/number

    avg = np.mat(avg)

    data = np.array(data)

    f = [[0 for x in range(m)] for y in range(m)]                

    g = [[0 for x in range(n)] for y in range(n)]

    f = np.mat(f)

    g = np.mat(g)

    # f=row-row and g = column column covariance matrices
    for x in range(25):

        temp = data[x]

        temp = np.mat(temp)


        f = f + (np.mat(temp-avg))*(np.transpose(np.mat(temp-avg)))

        g = g + (np.transpose(np.mat(temp-avg)))*(np.mat(temp-avg))

    f = f/25

    g = g/25    

    # get eigenVec and Eigen Values
    feignV,feignVec = np.linalg.eig(f)
    geignV,geignVec = np.linalg.eig(g)

    # choose top 5 and 3 most prominent eigen vectors
    U = feignVec[0:m,0:5]
    V = geignVec[0:n,0:3]

    U = np.transpose(U)

    M = [[[0 for x in range(5)] for y in range(5)] for z in range(25)]

    M = np.array(M)

    # map training data to new dimensions
    for x in range(25):

        temp = data[x]

        temp = np.mat(temp)

        temp = (U*temp)*V

        M[x] = np.array(temp)

    return M,U,V


def read(newData):

    data = [[] for x in range(25)]
    
    # read training data
    ultimate = open("ulti.txt","r")

    count=0
    
    for filename in ultimate:
        filename = filename.rstrip()
        # print filename
        f = open(filename,"r")

        for x in f:
            x=x.rstrip()
            x=x.split()
            data[count].append([int(x[0]),int(x[1]),int(x[2]),int(x[3]),int(x[4]),int(x[5]),float(x[6]),float(x[7])])

        count=count+1
        f.close()

    # apply 2d SVD
    M,U,V = train(data,25,8,25)

    # print M[1].shape

    temp = np.mat(newData)

    temp = (U*temp)*V

    ind=0

    mini=150000000000000

    # apply knn
    for x in range(25):

        dist = distance(M[x],temp,5,5)

        # print dist
        if(dist<mini):
            mini=dist
            ind=x

    print mini

    #classify
    if(ind<5 ): print "Foldedhands"
    elif(ind>=5 and ind<10 ): print "Handface"
    elif(ind>=10 and ind<15): print "Pointing"
    elif(ind<20): print "Beginning to end" 
    else: print "Clapping" 
    # print ind,mini

# Starting the camera/video capture
# if (len(sys.argv)>1): camera = cv2.VideoCapture(sys.argv[1])
# else: 
camera= cv2.VideoCapture(0)

if(len(sys.argv)>1):
    data1 = open(sys.argv[1],"w")
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
while count<25:
    #run()
    # Reading from capture
    flag1=flag2=fx=fy=0
    ret, image = camera.read()

    if image is None:
        print "Video Over"
        break
    # Applying background Subtraction
    fgmask = fgbg.apply(image.copy())
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    OriginalImg = image.copy()
    NoFilterImg = image.copy()

    image = cv2.blur(image,(5,5)) #try blurring with different kernel sizes to check effectiveness

    # skin detection using global HSV range
    """OpenCV uses different ranges for HSV as compared to other applications. h:0-130 s:0-255 v:0-255 
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


    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        fx = x+w/2
        fy= y+h/2
    
    # Show filteres skin image
    # cv2.imshow("skinimage",filterImg)

    
    # Taking and of skin image and foreground image
    thresh = cv2.bitwise_and(filterImg,fgmask)

    # thresh = cv2.erode(thresh,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))) #eroding the image

    thresh = cv2.dilate(thresh,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))) #dilating the image

    ret,thresh = cv2.threshold(thresh,50,255,cv2.THRESH_BINARY)

    (x,y) = np.array(thresh).shape

    # print fy
    if(fy==0):
        thresh1 = thresh[0:x,0:y/2]
        thresh2 = thresh[0:x,y/2:y] 
    else :
        thresh1 = thresh[0:x,0:fy]
        thresh2 = thresh[0:x,fy:y] 

    #getting all the contours
    a,contours, heirarchy = cv2.findContours(thresh1.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    max_area1=-1
    max_contour1 = None
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > max_area1:
            # fit and draw ellipse
            max_contour1 = c
            max_area1 = area

    a,contours, heirarchy = cv2.findContours(thresh2.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    max_area2=-1
    max_contour2 = None
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > max_area2:
            # fit and draw ellipse
            max_contour2 = c
            max_area2 = area

    if(max_area1>1000):
        ellipse1 = cv2.fitEllipse(max_contour1)
        # cv2.ellipse(OriginalImg,ellipse1,(0,255,0),2)
        M = cv2.moments(max_contour1)
        cx1 = int(M['m10']/M['m00'])
        cy1 = int(M['m01']/M['m00'])
        # mssg = str(cx) + " " + str(cy)+"\n"
        # data.write(mssg)

    else: 
        flag1=-1
        cx1=10**9
        cy1=10**9
        ellipse1 = [10**9,10**9,10**9]

    if(max_area2>1000):
        ellipse2 = cv2.fitEllipse(max_contour2) 
        # cv2.ellipse(OriginalImg,ellipse2,(0,255,0),2)
        M = cv2.moments(max_contour2)
        cx2 = int(M['m10']/M['m00']) 
        cy2 = int(M['m01']/M['m00']) + fy
        

    else:
        flag2=-1
        cx2=10**9
        cy2=10**9
        ellipse2 = [10**9,10**9,10**9]        

    if((flag1!=-1 or flag2 !=-1) and (fx!=0 and fy!=0)):
        # mssg = str(cx1) + " " + str(cy1)+" "+str(fx) + " " + str(fy)+" "+str(cx2) + " " + str(cy2)+" "+str(ellipse1[2])+" "+str(ellipse2[2])+"\n"
        # data1.write(mssg)
        mssg = [cx1,cy1,fx,fy,cx2,cy2,ellipse1[2],ellipse2[2]]
        data.append(mssg)
        count=count+1
    if(count==25):
        read(np.array(data))
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