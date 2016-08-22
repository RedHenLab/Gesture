import os
import cv2
import numpy as np
from PIL import Image


class CKDataLoader(object):

    def __init__(self,data_dir,label_dir):
        self.count_vid=0
        self.data_dir=data_dir
        self.count_sub=0
        self.label_dir=label_dir

        self.allowed_len=12

        self.sub_dirs=os.listdir(data_dir)[1:]
        #print self.sub_dirs
        self.updateDirs()


    def updateDirs(self):
#        print self.count_vid
        self.subject_dir=self.sub_dirs[self.count_sub]
        self.subject_vids=os.listdir(self.data_dir+"/"+self.subject_dir)
        if self.subject_vids[0].startswith("."):
            self.subject_vids=self.subject_vids[1:]
        #print self.subject_vids
        self.sub_vid_dir=os.listdir(self.data_dir+"/"+self.subject_dir+"/"+self.subject_vids[self.count_vid])


    def getLabel(self):
        file_path=self.label_dir+"/"+self.subject_dir+"/"+self.subject_vids[self.count_vid]
        dir_files=os.listdir(file_path)

        if len(dir_files)==0:
            return -1

        if dir_files[0].startswith("."):
            dir_files=dir_files[1:]
            if len(dir_files)==0:
                return -1

        print dir_files
        f=open(file_path+"/"+dir_files[0])
        label=f.readline()
        #print label[:-1][3]
        return int(label[:-1][3])


    def getSeqs(self,size,ret_same=False,get_lab=False):
        seqs=[]
        labels=[]

        if ret_same:
            if not get_lab:
                return self.cached_frames
            else:
                return self.cached_frames,self.cached_labels


        for i in range(size):
            file_path=self.data_dir+"/"+self.subject_dir+"/"+self.subject_vids[self.count_vid]
            vid_files=os.listdir(file_path)

            if vid_files[0].startswith("."):
                vid_files=vid_files[1:]

            file2_open=self.processSeq(vid_files)
            #print file2_open
            frames=[]

            for img in file2_open:
                #print file_path+"/"+img
                frame=cv2.imread(file_path+"/"+img,cv2.IMREAD_COLOR)
                frame=cv2.resize(frame,(145,145))
                #frame= Image.open(file_path+"/"+img)
                #cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
                #print frame.shape

                in_ = np.array(frame, dtype=np.float32)

                in_ = in_[:,:,::-1]
                in_ -= np.array((104.00698793,116.66876762,122.67891434))
                in_ = in_.transpose((2,0,1))
                #print in_.shape


                frames.append(in_)

            label=self.getLabel()

            self.count_vid+=1
            if self.count_vid>=len(self.subject_vids):
                self.count_vid=0
                self.count_sub+=1

            self.updateDirs()

            allowed_len=self.allowed_len
            if len(frames)<allowed_len:
                print "aww..."
                if not get_lab:
                    frames=self.getSeqs(1,ret_same,get_lab)[0]
                else:
                    frames,label=self.getSeqs(1,ret_same,get_lab)
                    frames=frames[0]
                    label=label[0]

            seqs.append(frames)
            labels.append(label)

        self.cached_frames=np.array(seqs)
        self.cached_labels=np.array(labels)

        if not get_lab:
            return np.array(seqs)#,labels
        else:
            return np.array(seqs),np.array(labels)


    def processSeq(self,seq):
        allowed_len=self.allowed_len
        #print "processing"

        if len(seq)>=allowed_len:
            num_extra=len(seq)-allowed_len

            num2del_before=num_extra/2+num_extra%2
            num2del_after=num_extra/2

            return seq[num2del_before:len(seq)-num2del_after]
        return seq


    def reset(self):
        self.count_vid=0
        self.count_sub=0
        self.updateDirs()


def processLabels(labels):
    prs_labels=[]
    for label in labels:
        prs_label=np.zeros((1,8,1,1))

        if not label==-1:
            prs_label[0,label-1,0,0]=1

        prs_labels.append(prs_label)

    return np.array(prs_labels)



if __name__=="__main__":
    #print os.listdir(".")
    data_dir="/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code_Theano/gesture_data/www.consortium.ri.cmu.edu/data/ck/CK+/cohn-kanade-images"
    label_dir="/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code_Theano/gesture_data/www.consortium.ri.cmu.edu/data/ck/CK+/Emotion"
    loader=CKDataLoader(data_dir,label_dir)
    print loader.getSeqs(1).shape
    for j in range(2):
        for i in range(400):
            frames,labels= loader.getSeqs(1,False,True)
            print frames.shape
        loader.reset()
        #print processLabels(labels).shape
