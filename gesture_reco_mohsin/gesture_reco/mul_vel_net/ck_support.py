import os
import cv2
import numpy as np

class CKDataLoader(object):

    def __init__(self,data_dir):
        self.count_vid=0
        self.count_sub=0
        self.sub_dirs=os.listdir(data_dir)[1:]
        #print self.sub_dirs
        self.updateDirs()


    def updateDirs(self):
        self.subject_dir=self.sub_dirs[self.count_sub]
        self.subject_vids=os.listdir(data_dir+"/"+self.subject_dir)[1:]
        self.sub_vid_dir=os.listdir(data_dir+"/"+self.subject_dir+"/"+self.subject_vids[self.count_vid])


    def getSeqs(self,size):
        seqs=[]
        labels=[]
        for i in range(size):
            file_path=data_dir+"/"+self.subject_dir+"/"+self.subject_vids[self.count_vid]
            vid_files=os.listdir(file_path)
            file2_open=self.processSeq(vid_files)
            frames=[]

            for img in file2_open:
                #print file_path+"/"+img
                frame=cv2.imread(file_path+"/"+img,cv2.IMREAD_COLOR)

                in_ = np.array(frame, dtype=np.float64)
                in_ = in_[:,:,::-1]
                in_ -= np.array((104.00698793,116.66876762,122.67891434))
                in_ = in_.transpose((2,0,1))

                frames.append(in_)

            seqs.append(frame)

            self.count_vid+=1
            if self.count_vid>=len(self.subject_vids):
                self.count_vid=0
                self.count_sub+=1

            self.updateDirs()
        return seqs


    def processSeq(self,seq):
        if len(seq)>9:
            num_extra=len(seq)-9

            num2del_before=num_extra/2+num_extra%2
            num2del_after=num_extra/2

            return seq[num2del_before-1:len(seq)-num2del_after]
        return seq





if __name__=="__main__":
    #print os.listdir(".")
    data_dir="/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code_Theano/gesture_data/www.consortium.ri.cmu.edu/data/ck/CK+/cohn-kanade-images"
    loader=CKDataLoader(data_dir)
    loader.getSeqs(5)
