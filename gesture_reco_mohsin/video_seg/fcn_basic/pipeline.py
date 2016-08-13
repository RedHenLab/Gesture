import cv2
import numpy as np



def check_cascade(img_path):
    face_cascade = cv2.CascadeClassifier('/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




class PersonSeperator(object):

    def __init__(self,img_path,max_iter=1):
        self.face_cascade = cv2.CascadeClassifier('/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
        self.max_iter = max_iter
        self.img = cv2.imread(img_path)


    def getFaceCentre(self):
        img = self.img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        face_centres=[]
        for (x,y,w,h) in faces:
            face_centres.append([y+h,x+w/2])
            cv2.circle(img,(x,y),5,(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

        cv2.imshow('img',img)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return face_centres


    def clusterPersons(self,human_labels):
        img = self.img
        face_centres = self.getFaceCentre()
        print face_centres
        Km = KMeans(human_labels,face_centres)
        Km.compute(self.max_iter)
        self.Km=Km
        return Km.labels


    def plotCluster(self,human_labels,cluster_labels):
        img = self.img
        height=img.shape[0]
        width=img.shape[1]
        print img.shape
        blank_image = np.zeros((height+1,width+1,3), np.uint8)
        print blank_image.shape
        print len(human_labels)
        print len(cluster_labels)

        for i in range(len(self.Km.centroids)):
            print self.Km.centroids[i]
            cv2.circle(blank_image,(self.Km.centroids[i][1],self.Km.centroids[i][0]),5,(255,255,255),-1)


        for i in range(len(human_labels)):
            human_label=human_labels[i]
            if human_label[0]<height and human_label[1]<width:
                blank_image[human_label[0],human_label[1]][cluster_labels[i]]=255

        cv2.imshow("cluster",blank_image)
        cv2.waitKey(0)



class KMeans(object):

    def __init__(self,dataset,init_centroids):
        num_iter=0
        self.dataset = np.array(dataset)
        self.centroids = np.array(init_centroids)
        self.labels=np.zeros(len(dataset))


    def getLabels(self):
        for i in range(len(self.dataset)):
            data_pt = self.dataset[i]
            self.labels[i] = self.getLabel(data_pt)


    def getLabel(self,data_pt):
        label=0
        min_dist=100000000
        for i in range(len(self.centroids)):
            centroid=self.centroids[i]
            dist=np.linalg.norm(centroid-data_pt)
            if dist<min_dist:
                min_dist=dist
                label=i
        return label


    def reAssignCentroids(self):
        sum_cluster=np.zeros((len(self.centroids),len(self.dataset[0])))
        num_cluster=np.zeros(len(self.centroids))

        for i in range(len(self.dataset)):
            sum_cluster[self.labels[i]]+=self.dataset[i]
            num_cluster[self.labels[i]]+=1

        for i in range(len(self.centroids)):
            self.centroids[i]=sum_cluster[i]/num_cluster[i]


    def compute(self,max_iter):
        for i in range(max_iter):
            self.getLabels()
            self.reAssignCentroids()




if __name__=="__main__":
    img = cv2.imread('/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code_Theano/fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/JPEGImages/2007_001430.jpg')
    check_cascade('/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code_Theano/fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/JPEGImages/2007_004000.jpg')
