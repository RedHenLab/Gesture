import numpy
import pylab
from PIL import Image
import pickle

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

from layers import *

class DeConvNet(object):

    def __init__(self,batch_size,num_features):
        self.batch_size=batch_size
        x=T.tensor4('x')
        self.x=x
        rng = numpy.random.RandomState(23455)

        poolsize=(2,2)

        convLayer1_input=x.reshape((self.batch_size,3,224,224))
        convLayer1_input_shape=(self.batch_size,3,224,224)
        convLayer1_filter=(num_features[0],3,3,3)
        self.convLayer1=PaddedConvLayer(rng,convLayer1_input,convLayer1_input_shape,convLayer1_filter)

        #batch_norm_layer1_input=self.convLayer1.output
        batch_norm_layer1_input=x.reshape((self.batch_size,3,224,224))
#        self.batch_norm_layer1=CNNBatchNormLayer(batch_norm_layer1_input,num_features[0])
        self.batch_norm_layer1=CNNBatchNormLayer(batch_norm_layer1_input,3)

    def test(self,test_set_x):
        out=self.batch_norm_layer1.output

        # Code for testing batch convolution
        #out_mean=self.batch_norm_layer1.batch_mean
        #out_var=self.batch_norm_layer1.batch_var
        #out_gamma=self.batch_norm_layer1.gamma

        index = T.lscalar()
        testDataX=theano.shared(test_set_x)

        batch_size=self.batch_size

        testDeConvNet=theano.function(
            inputs=[index],
            outputs=[out],
            on_unused_input='warn',
            givens={
                self.x :testDataX[index * batch_size: (index + 1) * batch_size]
            },
        )

        outs=[]

        n_test_batches=int(numpy.floor(len(test_set_x)/batch_size))
        print n_test_batches
        for batch_index in range(n_test_batches):
            out=testDeConvNet(batch_index)
            #print out
            for sam_out in out[0]:
#                 print sam_out
                outs.append(sam_out)


        #print outs
        return outs


def loadData():
    images=[]

    for i in range(1,10):
        path_str='/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code/DeConvNet/data/VOC2012/VOC_OBJECT/dataset_multlabel/images/2008_001563_0'+str(i)+'.png'
        img = Image.open(open(path_str))
        img=img.resize((224,224))
        img = numpy.asarray(img, dtype='float64') / 256.0
        img_ = img.transpose(2, 0, 1).reshape( 3, 224, 224)
        images.append(img_)
    return numpy.array(images)

if __name__=="__main__":
    deNet=DeConvNet(3,[64,64])
    numpy.set_printoptions(threshold='nan')
    print "loading data"
    data=loadData()
    print "finished loading data"

    outs=deNet.test(data)
    #print data[0]
    #print data[1]

    print "outs"
    print len(outs)
    print outs[1].shape
    print outs[1]
