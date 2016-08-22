import cv2
import numpy as np
from video_support import *



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
    cv2.imwrite("face_detect.png",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



class PersonFeeder(object):
    """
    This algorithm works for a video. It clusters the persons
    in each frame and also keeps a track of them using the algorihtm
    presented in the report.
    """

    def __init__(self,diff_seqs):
        """
        As there might be case in between that the face is not tracked properly,
        so we allow some buffer frames that are not tracked properly.
        """
        self.persons_data=[]
        self.diff_seqs = diff_seqs
        self.num_seq=0
        self.persons_center =[]
        self.between_jump=0


    def readImg(self,new_img_path,new_img):
        """
        The image can be passed using the path or the raw image itself
        """
        if not img_path==None:
            self.img = cv2.imread(img_path)
            return

        if img == None:
            raise TypeError("No input")

        self.img = img


    def CheckIfReset(self):
        """
        Reset is done with the assumption that the same persons are
        no longer present in the video anymore.
        """
        if self.between_jump >= self.diff_seqs:
            print "had to reset"
            self.num_seq = 0
            self.persons_center =[]
            self.persons_data = []
            self.between_jump = 0


    def checkProceed(self,human_labels,new_face_centers):
        """
        Check if the current frame is proper ie the face is detected and
        the human labels are properly segmented.
        """

        if len(new_face_centers) ==0:
            self.between_jump +=1
            return False

        if len(human_labels) ==0:
            self.between_jump+=1
            return False

        return True


    def track(self,human_labels,new_img_path=None,new_img=None):
        """
        Detect the face and check if the frame is proper and cluster
        the persons out.
        """
        self.person_sep = PersonSeperator(new_img_path,new_img)
        new_face_centers = self.person_sep.getFaceCentre()

        if not self.checkProceed(human_labels,new_face_centers):
            print "failed proceed"
            return

        if len(self.persons_center) == 0:
            print "zero len persons_center"
            for face_center in new_face_centers:
                self.persons_center.append([face_center])

        self.weighted_centers = self.getWeightedCenters()

        if not len(new_face_centers) == len(self.weighted_centers):
            print "not equal len of weighted and new face centres"
            self.between_jump+=1
        else:
            self.getAndAssignData(new_face_centers,human_labels)

        self.CheckIfReset()


    def plotNewImages(self):
        """
        Code to plot the images of the seperated persons
        """
        print "num_persons" + str(len(self.persons_data))
        for i in range(len(self.persons_data)):
            person_data = self.persons_data[i]
            for image in person_data:
                cv2.imshow("cluster "+str(i),image)
                cv2.waitKey(33)

        if len(self.persons_data)>0:
            #cv2.waitKey(0)
            cv2.destroyAllWindows()


    def saveFrames(self):
        """
        Save the video for each of the user.
        """
        for i in range(len(self.persons_data)):
            person_data = self.persons_data[i]
            saveRawVideo(person_data,"out_"+str(i))


    def getAndAssignData(self,new_face_centers,human_labels):
        """
        This function gets the data and saves in the list that will be
        used in the further steps.
        """
        self.assignFaceCentres(new_face_centers,self.weighted_centers)
        self.num_seq+=1

        cluster_labels = self.person_sep.clusterPersons(human_labels)
        sep_images = self.person_sep.createNewImages(human_labels,cluster_labels,len(new_face_centers))

        for i in range(len(new_face_centers)):
            label_index = int(self.new_face_labels[i])
            (self.persons_center[label_index]).append(new_face_centers[i])

            if label_index >= len(self.persons_data):
                self.persons_data.append([sep_images[i]])
            else:
                (self.persons_data[label_index]).append(sep_images[i])


    def assignFaceCentres(self,new_face_centers,weighted_centers):
        """
        Given a list of new face centres, this function looks at the history
        and assign the face to the cluster which is nearest. The weighted center
        for each cluster using history with the recent frames getting more weight.
        """
        self.new_face_labels = np.zeros(len(new_face_centers))
        for i in range(len(new_face_centers)):
            face_center = new_face_centers[i]
            min_index = 0
            min_dist = 100000000

            for j in range(len(weighted_centers)):
                weighted_center = weighted_centers[j][0]
                dist=np.linalg.norm(weighted_center-face_center)
                if dist < min_dist:
                    min_dist = dist
                    min_index = j

            self.new_face_labels[i] = min_index


    def getWeightedCenters(self):
        """
        Computes weighted center for each cluster using the history.
        The weights are reduced linearly with the least weight =1 and
        the most recent frame getting the highest
        """
        weights = np.zeros((1,len(self.persons_center[0])))
        weighted_sum = 0

        for i in range(len(self.persons_center[0])):
            weights[0][i] = i+1
            weighted_sum+=i+1

        weighted_center = np.zeros((len(self.persons_center),1,2))

        for i in range(len(self.persons_center)):
            person_centers = self.persons_center[i]
            weighted_center[i][0] = np.dot(weights,person_centers) / weighted_sum

        return weighted_center



class PersonSeperator(object):
    """
    Given a image, it first computes the face centre
    using Viola and Jones descriptor and then cluster out
    the persons using K-means clustering
    """

    def __init__(self,img_path=None,img=None,max_iter=1):
        self.face_cascade = cv2.CascadeClassifier('/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
        self.max_iter = max_iter

        if not img_path==None:
            self.img = cv2.imread(img_path)
            return

        if img == None:
            raise TypeError("No input")

        self.img = img


    def updateImagePath(self,img_path):
        self.img = cv2.imread(img_path)


    def updateImage(self,img):
        self.img = img


    def getFaceCentre(self):
        """
        Computes the face center for all the faces in the image.
        """
        img = self.img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        face_centres=[]
        for (x,y,w,h) in faces:
            face_centres.append([y+h,x+w/2])
            cv2.circle(img,(x,y),5,(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

        cv2.destroyAllWindows()

        return face_centres


    def clusterPersons(self,human_labels):
        """
        Computes face centres and pass them to clustering algorithm
        to get the cluster out all the persons.
        """
        img = self.img
        face_centres = self.getFaceCentre()
        Km = KMeans(human_labels,face_centres)
        Km.compute(self.max_iter)
        self.Km=Km
        return Km.labels


    def createNewImages(self,human_labels,cluster_labels,num_persons):
        """
        Creates images for each person in the image with just the data for that
        particular individual.
        """
        img = self.img
        height=img.shape[0]
        width=img.shape[1]
        blank_images=[]

        for i in range(num_persons):
            blank_image = np.zeros((height+1,width+1,3), np.uint8)
            blank_images.append(blank_image)

        for j in range(len(human_labels)):
            person_label = cluster_labels[j]
            blank_image = blank_images[int(person_label)]
            x = human_labels[j][0]
            y = human_labels[j][1]
            if x < height and y<width:
                blank_image[x,y] = img[x,y]

        return blank_images


    def plotCluster(self,human_labels,cluster_labels):
        """
        This code is to debug how the cluster and its boundary are
        """
        img = self.img
        height=img.shape[0]
        width=img.shape[1]
        blank_image = np.zeros((height+1,width+1,3), np.uint8)

        for i in range(len(self.Km.centroids)):
            cv2.circle(blank_image,(self.Km.centroids[i][1],self.Km.centroids[i][0]),5,(255,255,255),-1)


        for i in range(len(human_labels)):
            human_label=human_labels[i]
            if human_label[0]<height and human_label[1]<width:
                blank_image[human_label[0],human_label[1]][cluster_labels[i]]=255

        cv2.imshow("cluster",blank_image)
        cv2.waitKey(0)


class KMeans(object):
    """
    The K-means clustering algorithm
    """

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
    im_path = '/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code_Theano/fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_002103.jpg'
    check_cascade(im_path)
