import sys
try:
    import caffe
except ImportError:
    print "trial 1"
import caffe
import numpy as np
import pickle

from caffe.proto import caffe_pb2

params = caffe_pb2.NetParameter()
f=open("DeconvNet_trainval_inference.caffemodel",'rb')

params.ParseFromString(f.read())

net_layers=params.layers
convW=[]

# getting the weights for convolution
# Tested by copy pasting in terminal
layer_num=0
for layer in net_layers:
    layer_num=layer_num+1

    if layer.type==4:
        raw_dataW=layer.blobs[0].data
        w_blob=layer.blobs[0].num
        W_reshaped=np.array(raw_dataW).reshape(w_blob.num,w_blob.channels,w_blob.height,w_blob.height)
        pickle.dump(W_reshaped,open("weights/weights"+str(layer_num)+".p"))
