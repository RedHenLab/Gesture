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



class PersonFeeder(object):

    def __init__(self,diff_seqs):
        self.persons_data=[]
        self.diff_seqs = diff_seqs
        self.num_seq=0
        self.persons_center =[]


    def readImg(self,new_img_path,new_img):
        if not img_path==None:
            self.img = cv2.imread(img_path)
            return

        if img == None:
            raise TypeError("No input")

        self.img = img


    def track(self,human_labels,new_img_path=None,new_img=None):
        self.person_sep = PersonSeperator(new_img_path,new_img)
        new_face_centers = self.person_sep.getFaceCentre()

    #    self.readImg(new_img_path,img)

        if len(self.persons_center) == 0:
            self.persons_center = new_face_centers
    #        self.__genPersonsData()

        self.weighted_centers = self.getWeightedCenters()

        if not len(new_face_centers) == len(self.weighted_centers):
            self.num_seq = 0
            self.persons_center =[]
            self.persons_data = []
        else:
            self.getAndAssignData(new_face_centers,human_labels)


    """
    def __genPersonsData(self):
        for i in range(len(self.persons_center)):
    """


    def plotNewImages(self):
        for person_data in self.persons_data:
            for image in person_data:
                cv2.imshow("cluster",image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()



    def getAndAssignData(self,new_face_centers,human_labels):
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
        self.new_face_labels = np.zeros(len(new_face_centers))
        for i in range(len(new_face_centers)):
            face_center = new_face_centers[i]
            min_index = 0
            min_dist = 100000000

            for j in range(len(weighted_centers)):
                weighted_center = weighted_centers[j]
                dist=np.linalg.norm(weighted_center-face_center)
                if dist < min_dist:
                    min_dist = dist
                    min_index = j

            self.new_face_labels[i] = j


    def getWeightedCenters(self):
        weights = np.zeros((len(self.persons_center[0]),1))
        weighted_sum = 0
        for i in range(len(self.persons_center[0])):
            weights[i][0] = i
            weighted_sum+=i
        return weights * self.persons_center / weighted_sum




class PersonSeperator(object):

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


    def createNewImages(self,human_labels,cluster_labels,num_persons):
        img = self.img
        print img.shape
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
