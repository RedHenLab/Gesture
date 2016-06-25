import cv2
import numpy as np

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
    #out = cv2.VideoWriter('output.avi',-1, 5.0, (width,height))


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


if __name__=="__main__":
    #loadVideo("/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/news_data/test/news_01.mp4")
    #saveImage(0)

    loadVideo("output.avi")
