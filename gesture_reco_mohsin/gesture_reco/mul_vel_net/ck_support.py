import os
import cv2
import numpy as np
from PIL import Image

class CKDataLoader(object):
    """
    As the video is in the sequence of image format and the labeled
    and the unlabeled sequences are not properly seperated. So This
    class helps in reading the labels, remove the images in the sequence
    so that in the end each sequence has fixed number of frames.


    This works by listing the directories of the user and the
    sub directories of the sequences for each user. Each time a new
    request comes up, the count in the sub dir is moved by one and if all
    the sequences of the user are over then the user directory count
    is updated.

    The output has to be of the 1-of-k format so that is also processed
    """

    def __init__(self,data_dir,label_dir):
        self.count_vid=0
        self.data_dir=data_dir
        self.count_sub=0
        self.label_dir=label_dir

        self.allowed_len=12

        self.sub_dirs=os.listdir(data_dir)[1:]
        self.updateDirs()


    def updateDirs(self):
        """
        Once the counts are updated, the directories are updated.
        """
        self.subject_dir=self.sub_dirs[self.count_sub]
        self.subject_vids=os.listdir(self.data_dir+"/"+self.subject_dir)
        if self.subject_vids[0].startswith("."):
            self.subject_vids=self.subject_vids[1:]
        self.sub_vid_dir=os.listdir(self.data_dir+"/"+self.subject_dir+"/"+self.subject_vids[self.count_vid])


    def getLabel(self):
        """
        Gets a label for the sequence. If the sequence is not labeled, the directory
        will be missing, hence that is given label 0 at all representation. This will be used
        in the training procedure. It will make the weight for the supervised part 0.
        """
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
        return int(label[:-1][3])


    def getSeqs(self,size,ret_same=False,get_lab=False):
        """
        Reads the images from the file. If the number of sequence is less than
        the allowed sequence, then that sequence is ignored. Else the frames at
        both the ends are removed so that left number of sequence are the same
        as the allowed seq. Also updates the pointers accordingly for the directories
        providing the data.

        ret_same True does not get a new sequence, instead it gives a cached
        sequence.
        """
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
            frames=[]

            for img in file2_open:
                frame=cv2.imread(file_path+"/"+img,cv2.IMREAD_COLOR)
                frame=cv2.resize(frame,(145,145))

                in_ = np.array(frame, dtype=np.float32)

                in_ = in_[:,:,::-1]
                in_ -= np.array((104.00698793,116.66876762,122.67891434))
                in_ = in_.transpose((2,0,1))

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
        """
        Remove the sequence at the corners.
        """
        allowed_len=self.allowed_len

        if len(seq)>=allowed_len:
            num_extra=len(seq)-allowed_len

            num2del_before=num_extra/2+num_extra%2
            num2del_after=num_extra/2

            return seq[num2del_before:len(seq)-num2del_after]
        return seq


    def reset(self):
        """
        Reset the directory pointers.
        """
        self.count_vid=0
        self.count_sub=0
        self.updateDirs()


def processLabels(labels):
    """
    Convert the date from a number to 1 of K representation and
    0 at all if not labeled.
    """
    prs_labels=[]
    for label in labels:
        prs_label=np.zeros((1,8,1,1))

        if not label==-1:
            prs_label[0,label-1,0,0]=1

        prs_labels.append(prs_label)

    return np.array(prs_labels)



if __name__=="__main__":
    data_dir="/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code_Theano/gesture_data/www.consortium.ri.cmu.edu/data/ck/CK+/cohn-kanade-images"
    label_dir="/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code_Theano/gesture_data/www.consortium.ri.cmu.edu/data/ck/CK+/Emotion"
    loader=CKDataLoader(data_dir,label_dir)
    print loader.getSeqs(1).shape
    for j in range(2):
        for i in range(400):
            frames,labels= loader.getSeqs(1,False,True)
            print frames.shape
        loader.reset()
