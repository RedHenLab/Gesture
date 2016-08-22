import cv2
import numpy as np

def loadVideo(file_name,max_frames=10**10):
    """
    Load a video file and do a mean subtraction from it to get.
    Process all the frames or do only till the maximum number of frames.
    """
    cap=cv2.VideoCapture(file_name)
    frames_np=[]
    frame_num=0

    while(cap.isOpened()):
        ret,frame=cap.read()

        if not ret:
            break

        in_ = np.array(frame, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_ = in_.transpose((2,0,1))

        frames_np.append(in_)

        frame_num+=1
        if frame_num>max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()

    print len(frames_np)

    return frames_np



def loadRawVideo(file_name,max_frames=10**10):
    """
    Same as above but do not do mean subtraction
    """
    cap=cv2.VideoCapture(file_name)
    frames_np=[]
    frame_num=0

    while(cap.isOpened()):
        ret,frame=cap.read()

        if not ret:
            break

        frames_np.append(frame)

        frame_num+=1
        if frame_num>max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()

    print len(frames_np)

    return frames_np



def saveImage(data,class_label):
    """
    Given the segmentation labels, it shows only the data for the
    pixels classified as a given class label. It also saves the output.
    """
    height=data.shape[0]
    width=data.shape[1]
    blank_image = np.zeros((height,width,3), np.uint8)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j]==class_label:
                #print "hello"
                blank_image[i][j][0]=255

    cv2.imshow("test",blank_image)
    cv2.imwrite("fcn_out.png",blank_image)


def saveRawVideo(data,name):
    """
    Saves the video with only the frames given in the data.
    """
    height=data[0].shape[0]
    width=data[1].shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(name+'.mp4',fourcc, 20.0, (width,height))

    for frame in data:
        out.write(frame)
        print "saving frame"

    out.release()



def saveVideo(data,class_label):
    """
    Given the raw segmentation labels, it only saves the
    data for the given class label as an image.
    """
    height=data[0].shape[0]
    width=data[1].shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4',fourcc, 33.0, (width,height))

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


def genLabelData(data,class_label,out_shape):
    """
    This function generates a list of pixel coordinates with
    the pixel classified as a given class label
    """
    height=data.shape[0]
    width=data.shape[1]
    human_labels=[]
    height_diff = np.ceil((out_shape[2]-height)/2)
    width_diff = np.ceil((out_shape[3]-width)/2)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j]==class_label:
                #print "hello"
                human_labels.append([i-height_diff,j-width_diff])

    return human_labels



if __name__=="__main__":
    #loadVideo("/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/news_data/test/news_01.mp4")
    #saveImage(0)

    loadVideo("output.avi")
