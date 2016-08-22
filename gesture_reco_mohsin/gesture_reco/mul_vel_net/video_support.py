import cv2
import numpy as np
#from layers import *
import theano

def loadVideo(file_name,max_frames=10**10):
    cap=cv2.VideoCapture(file_name)
    frames_np=[]
    frame_num=0

    while(cap.isOpened()):
        ret,frame=cap.read()

        if not ret:
            break

        in_ = np.array(frame, dtype=np.float64)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_ = in_.transpose((2,0,1))
        #print in_.shape

        frames_np.append(in_)

        # code to show the video
        #
        #cv2.imshow('frame',frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        frame_num+=1
        if frame_num>max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()

    print len(frames_np)

    return frames_np


def saveImage(data,class_label):
    height=data.shape[0]
    width=data.shape[1]
    blank_image = np.zeros((height,width,3), np.uint8)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j]==class_label:
                #print "hello"
                blank_image[i][j][0]=255

    cv2.imshow("test",blank_image)
    cv2.waitKey(0)


def saveVideo(data,class_label):
    height=data[0].shape[0]
    width=data[1].shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (width,height))

    for frame in data:
        print "saving frame"
        blank_image = np.zeros((height,width,3), np.uint8)
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if frame[i][j]==class_label:
                    #print "hello"
                    blank_image[i][j][0]=255
        out.write(blank_image)

    out.release()



# the input data is 3D vector with the zeroth dimension represnting
# the index of frame and
def sampleVideo(data,factor,num_new_frames=0):
    n=len(data)-1
    n_r=num_new_frames
    if num_new_frames==0:
        n_r=n

    T_mat=np.zeros((4*n,n+1))
    A_mat=np.zeros((4*n,4*n))

    for i in range(n):
        T_mat[4*i,i]=1
        T_mat[4*i+1,i+1]=1

        A_mat[4*i,4*i:4*i+4]=[1,0,0,0]      # @x_k
        A_mat[4*i+1,4*i:4*i+4]=[1,1,1,1]    # @x_{k+1}
        if not i==n-1:
            A_mat[4*i+2,4*i:4*i+8]=[0,1,2,3,0,-1,0,0]     #S_k'(x_{k+1})=S_{k+1}'(x_{k+1})
            A_mat[4*i+3,4*i:4*i+8]=[0,0,2,6,0,0,-2,0]      #S_k''(x_{k+1})=S_{k+1}''(x_{k+1})

    A_mat[4*n-2,0:4]=[0,0,2,0]
    A_mat[4*n-1,4*n-4:4*n]=[0,0,2,6]

    A_inv=np.linalg.inv(A_mat)
    p_vec=np.dot(A_inv,T_mat)

    u_s=range(n_r+1)
    R_mat=np.zeros((n_r+1,4*n))

    for i in range(n_r+1):
        u_s[i]=i*factor
        p=np.floor(u_s[i])
        x_p=p

        for h in range(4):
            R_mat[i,4*p+h]=(u_s[i]-x_p)**h

    W=np.dot(R_mat,p_vec)
    return W



if __name__=="__main__":
    #loadVideo("/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/news_data/test/news_01.mp4")
    #saveImage(0)

    #loadVideo("output.avi")

    W=sampleVideo(range(3),0.99999)
    print np.dot(W,[0,3,9])
