import numpy as np
import numpy.ceil as cl

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

from layers import *

class FCN(object):

    def __init__(self,batch_size,input_size):
        self.batch_size=batch_size
        poolsize=(2,2)

        x=T.tensor4('x')
        self.x=x
        rng = numpy.random.RandomState(23455)

        convLayer1_input=x.reshape((self.batch_size,3,input_size[0],input_size[1]))
        convLayer1_input_shape=(self.batch_size,3,input_size[0],input_size[1])
        convLayer1_filter=(64,3,3,3)
        conv1_pad=100
        weights_conv1_1=pickle.load(open("weights/conv1_1W.p","rb"))
        bias_conv1_1=pickle.load(open("weights/bias1_1.p","rb"))
        self.convLayer1_1=PaddedConvLayer(rng,convLayer1_input,convLayer1_input_shape,convLayer1_filter,conv1_pad)
        self.convLayer1_1.assignParams(weights_conv1_1,bias_conv1_1)

        relu_layer1_1_input=self.convLayer1_1.output
        self.relu_layer1_1=ReLuLayer(relu_layer1_1_input)

        convLayer1_2_input=self.relu_layer1_1.output
        convLayer1_2_input_shape=(self.batch_size,64,input_size[0]+198,input_size[1]+198)
        convLayer1_2_filter=(64,64,3,3)
        conv1_2pad=1
        weights_conv1_2=pickle.load(open("weights/conv1_2W.p","rb"))
        bias_conv1_2=pickle.load(open("weights/bias1_2.p","rb"))
        self.convLayer1_2=PaddedConvLayer(rng,convLayer1_2_input,convLayer1_2_input_shape,convLayer1_2_filter,conv1_2pad)
        self.convLayer1_2.assignParams(weights_conv1_2,bias_conv1_2)

        relu_layer1_2_input=self.convLayer1_2.output
        self.relu_layer1_2=ReLuLayer(relu_layer1_2_input)

        max_pool_layer1_input=self.relu_layer1_2.output
        self.max_pool_layer1=MaxPoolLayer(max_pool_layer1_input)


        convLayer2_1_input=self.max_pool_layer1.output
        convLayer2_1_input_shape=(self.batch_size,64,cl((input_size[0]+198)/2),cl((input_size[1]+198)/2))
        convLayer2_1_filter=(128,64,3,3)
        weights_conv2_1=pickle.load(open("weights/conv2_1W.p","rb"))
        bias_conv2_1=pickle.load(open("weights/bias2_1.p","rb"))
        conv2_1pad=1
        self.convLayer2_1=PaddedConvLayer(rng,convLayer2_1_input,convLayer2_1_input_shape,convLayer2_1_filter,conv2_1pad)
        self.convLayer2_1.assignParams(weights_conv2_1,bias_conv2_1)

        relu_layer2_1_input=self.convLayer2_1.output
        self.relu_layer2_1=ReLuLayer(relu_layer2_1_input)

        convLayer2_2_input=self.relu_layer2_1.output
        convLayer2_2_input_shape=(self.batch_size,128,cl((input_size[0]+198)/2),cl((input_size[1]+198)/2))
        convLayer2_2_filter=(128,128,3,3)
        weights_conv2_2=pickle.load(open("weights/conv2_2W.p","rb"))
        bias_conv2_2=pickle.load(open("weights/bias2_2.p","rb"))
        conv2_2pad=1
        self.convLayer2_2=PaddedConvLayer(rng,convLayer2_2_input,convLayer2_2_input_shape,convLayer2_2_filter,conv2_2pad)
        self.convLayer2_2.assignParams(weights_conv2_2,bias_conv2_2)

        relu_layer2_2_input=self.convLayer2_2.output
        self.relu_layer2_2=ReLuLayer(relu_layer2_2_input)

        max_pool_layer2_input=self.relu_layer2_2.output
        self.max_pool_layer2=MaxPoolLayer(max_pool_layer2_input,ignore_border_g=False)


        convLayer3_1_input=self.max_pool_layer2.output
        convLayer3_1_input_shape=(self.batch_size,128,cl(cl((input_size[0]+198)/2)/2),cl(cl((input_size[1]+198)/2)/2))
        convLayer3_1_filter=(256,128,3,3)
        weights_conv3_1=pickle.load(open("weights/conv3_1W.p","rb"))
        bias_conv3_1=pickle.load(open("weights/bias3_1.p","rb"))
        conv3_1pad=1
        self.convLayer3_1=PaddedConvLayer(rng,convLayer3_1_input,convLayer3_1_input_shape,convLayer3_1_filter,conv3_1pad)
        self.convLayer3_1.assignParams(weights_conv3_1,bias_conv3_1)

        relu_layer3_1_input=self.convLayer3_1.output
        self.relu_layer3_1=ReLuLayer(relu_layer3_1_input)

        convLayer3_2_input=self.relu_layer3_1.output
        convLayer3_2_input_shape=(self.batch_size,256,cl(cl((input_size[0]+198)/2)/2),cl(cl((input_size[1]+198)/2)/2))
        convLayer3_2_filter=(256,256,3,3)
        weights_conv3_2=pickle.load(open("weights/conv3_2W.p","rb"))
        bias_conv3_2=pickle.load(open("weights/bias3_2.p","rb"))
        conv3_2pad=1
        self.convLayer3_2=PaddedConvLayer(rng,convLayer3_2_input,convLayer3_2_input_shape,convLayer3_2_filter,conv3_2pad)
        self.convLayer3_2.assignParams(weights_conv3_2,bias_conv3_2)

        relu_layer3_2_input=self.convLayer3_2.output
        self.relu_layer3_2=ReLuLayer(relu_layer3_2_input)

        convLayer3_3_input=self.relu_layer3_2.output
        convLayer3_3_input_shape=(self.batch_size,256,cl(cl((input_size[0]+198)/2)/2),cl(cl((input_size[1]+198)/2)/2))
        convLayer3_3_filter=(256,256,3,3)
        weights_conv3_3=pickle.load(open("weights/conv3_3W.p","rb"))
        bias_conv3_3=pickle.load(open("weights/bias3_3.p","rb"))
        conv3_3pad=1
        self.convLayer3_3=PaddedConvLayer(rng,convLayer3_3_input,convLayer3_3_input_shape,convLayer3_3_filter,conv3_3pad)
        self.convLayer3_3.assignParams(weights_conv3_3,bias_conv3_3)

        relu_layer3_3_input=self.convLayer3_3.output
        self.relu_layer3_3=ReLuLayer(relu_layer3_3_input)

        max_pool_layer3_input=self.relu_layer3_3.output
        self.max_pool_layer3=MaxPoolLayer(max_pool_layer3_input,ignore_border_g=False)


        convLayer4_1_input=self.max_pool_layer3.output
        convLayer4_1_input_shape=(self.batch_size,256,cl(cl(cl((input_size[0]+198)/2)/2)/2),cl(cl(cl((input_size[1]+198)/2)/2)/2))
        convLayer4_1_filter=(512,256,3,3)
        weights_conv4_1=pickle.load(open("weights/conv4_1W.p","rb"))
        bias_conv4_1=pickle.load(open("weights/bias4_1.p","rb"))
        conv4_1pad=1
        self.convLayer4_1=PaddedConvLayer(rng,convLayer4_1_input,convLayer4_1_input_shape,convLayer4_1_filter,conv4_1pad)
        self.convLayer4_1.assignParams(weights_conv4_1,bias_conv4_1)

        relu_layer4_1_input=self.convLayer4_1.output
        self.relu_layer4_1=ReLuLayer(relu_layer4_1_input)

        convLayer4_2_input=self.relu_layer4_1.output
        convLayer4_2_input_shape=(self.batch_size,512,cl(cl(cl((input_size[0]+198)/2)/2)/2),cl(cl(cl((input_size[1]+198)/2)/2)/2))
        convLayer4_2_filter=(512,512,3,3)
        weights_conv4_2=pickle.load(open("weights/conv4_2W.p","rb"))
        bias_conv4_2=pickle.load(open("weights/bias4_2.p","rb"))
        conv4_2pad=1
        self.convLayer4_2=PaddedConvLayer(rng,convLayer4_2_input,convLayer4_2_input_shape,convLayer4_2_filter,conv4_2pad)
        self.convLayer4_2.assignParams(weights_conv4_2,bias_conv4_2)

        relu_layer4_2_input=self.convLayer4_2.output
        self.relu_layer4_2=ReLuLayer(relu_layer4_2_input)

        convLayer4_3_input=self.relu_layer4_2.output
        convLayer4_3_input_shape=(self.batch_size,512,cl(cl(cl((input_size[0]+198)/2)/2)/2),cl(cl(cl((input_size[1]+198)/2)/2)/2))
        convLayer4_3_filter=(512,512,3,3)
        weights_conv4_3=pickle.load(open("weights/conv4_3W.p","rb"))
        bias_conv4_3=pickle.load(open("weights/bias4_3.p","rb"))
        conv4_3pad=1
        self.convLayer4_3=PaddedConvLayer(rng,convLayer4_3_input,convLayer4_3_input_shape,convLayer4_3_filter,conv4_3pad)
        self.convLayer4_3.assignParams(weights_conv4_3,bias_conv4_3)

        relu_layer4_3_input=self.convLayer4_3.output
        self.relu_layer4_3=ReLuLayer(relu_layer4_3_input)

        max_pool_layer4_input=self.relu_layer4_3.output
        self.max_pool_layer4=MaxPoolLayer(max_pool_layer4_input,ignore_border_g=False)


        convLayer5_1_input=self.max_pool_layer4.output
        convLayer5_1_input_shape=(self.batch_size,512,cl(cl(cl(cl((input_size[0]+198)/2)/2)/2)/2),cl(cl(cl(cl((input_size[1]+198)/2)/2)/2)/2))
        convLayer5_1_filter=(512,512,3,3)
        weights_conv5_1=pickle.load(open("weights/conv5_1W.p","rb"))
        bias_conv5_1=pickle.load(open("weights/bias5_1.p","rb"))
        conv5_1pad=1
        self.convLayer5_1=PaddedConvLayer(rng,convLayer5_1_input,convLayer5_1_input_shape,convLayer5_1_filter,conv5_1pad)
        self.convLayer5_1.assignParams(weights_conv5_1,bias_conv5_1)

        relu_layer5_1_input=self.convLayer5_1.output
        self.relu_layer5_1=ReLuLayer(relu_layer5_1_input)

        convLayer5_2_input=self.relu_layer5_1.output
        convLayer5_2_input_shape=(self.batch_size,512,cl(cl(cl(cl((input_size[0]+198)/2)/2)/2)/2),cl(cl(cl(cl((input_size[1]+198)/2)/2)/2)/2))
        convLayer5_2_filter=(512,512,3,3)
        weights_conv5_2=pickle.load(open("weights/conv5_2W.p","rb"))
        bias_conv5_2=pickle.load(open("weights/bias5_2.p","rb"))
        conv5_2pad=1
        self.convLayer5_2=PaddedConvLayer(rng,convLayer5_2_input,convLayer5_2_input_shape,convLayer5_2_filter,conv5_2pad)
        self.convLayer5_2.assignParams(weights_conv5_2,bias_conv5_2)

        relu_layer5_2_input=self.convLayer5_2.output
        self.relu_layer5_2=ReLuLayer(relu_layer5_2_input)

        convLayer5_3_input=self.relu_layer5_2.output
        convLayer5_3_input_shape=(self.batch_size,512,cl(cl(cl(cl((input_size[0]+198)/2)/2)/2)/2),cl(cl(cl(cl((input_size[1]+198)/2)/2)/2)/2))
        convLayer5_3_filter=(512,512,3,3)
        weights_conv5_3=pickle.load(open("weights/conv5_3W.p","rb"))
        bias_conv5_3=pickle.load(open("weights/bias5_3.p","rb"))
        conv5_3pad=1
        self.convLayer5_3=PaddedConvLayer(rng,convLayer5_3_input,convLayer5_3_input_shape,convLayer5_3_filter,conv5_3pad)
        self.convLayer5_3.assignParams(weights_conv5_3,bias_conv5_3)

        relu_layer5_3_input=self.convLayer5_3.output
        self.relu_layer5_3=ReLuLayer(relu_layer5_3_input)

        max_pool_layer5_input=self.relu_layer5_3.output
        self.max_pool_layer5=MaxPoolLayer(max_pool_layer5_input,ignore_border_g=False)

        convLayer6_1_input=self.max_pool_layer5.output
        convLayer6_1_input_shape=(self.batch_size,512,cl(cl(cl(cl(cl((input_size[0]+198)/2)/2)/2)/2)/2),cl(cl(cl(cl(cl((input_size[1]+198)/2)/2)/2)/2)/2))
        convLayer6_1_filter=(4096,512,7,7)
        weights_conv6_1=pickle.load(open("weights/conv6_1W.p","rb"))
        bias_conv6_1=pickle.load(open("weights/bias6_1.p","rb"))
        conv6_1pad=0
        self.convLayer6_1=PaddedConvLayer(rng,convLayer6_1_input,convLayer6_1_input_shape,convLayer6_1_filter,conv6_1pad)
        self.convLayer6_1.assignParams(weights_conv6_1,bias_conv6_1)

        relu_layer6_1_input=self.convLayer6_1.output
        self.relu_layer6_1=ReLuLayer(relu_layer6_1_input)


        convLayer7_1_input=self.relu_layer6_1.output
        convLayer7_1_input_shape=(self.batch_size,4096,cl(cl(cl(cl(cl((input_size[0]+198)/2)/2)/2)/2)/2)-6,cl(cl(cl(cl(cl((input_size[1]+198)/2)/2)/2)/2)/2)-6)
        convLayer7_1_filter=(4096,4096,1,1)
        weights_conv7_1=pickle.load(open("weights/conv7_1W.p","rb"))
        bias_conv7_1=pickle.load(open("weights/bias7_1.p","rb"))
        conv7_1pad=0
        self.convLayer7_1=PaddedConvLayer(rng,convLayer7_1_input,convLayer7_1_input_shape,convLayer7_1_filter,conv7_1pad)
        self.convLayer7_1.assignParams(weights_conv7_1,bias_conv7_1)

        relu_layer7_1_input=self.convLayer7_1.output
        self.relu_layer7_1=ReLuLayer(relu_layer7_1_input)


        score_fr_Layer_input=self.relu_layer7_1.output
        score_fr_Layer_input_shape=(self.batch_size,4096,cl(cl(cl(cl(cl((input_size[0]+198)/2)/2)/2)/2)/2)-6,cl(cl(cl(cl(cl((input_size[1]+198)/2)/2)/2)/2)/2)-6)
        score_fr_Layer_filter=(21,4096,1,1)
        weights_score_fr=pickle.load(open("weights/score_fr_W.p","rb"))
        bias_score_fr=pickle.load(open("weights/bias_score_fr.p","rb"))
        score_fr_pad=0
        self.score_fr_Layer=PaddedConvLayer(rng,score_fr_Layer_input,score_fr_Layer_input_shape,score_fr_Layer_filter,score_fr_pad)
        self.score_fr_Layer.assignParams(weights_score_fr,bias_score_fr)

        upscore2_Layer_input=self.score_fr_Layer.output
        upscore2_Layer_input_shape=(self.batch_size,21,cl(cl(cl(cl(cl((input_size[0]+198)/2)/2)/2)/2)/2)-6,cl(cl(cl(cl(cl((input_size[1]+198)/2)/2)/2)/2)/2)-6)
        upscore2_Layer_filter=(21,21,4,4)
        upscore2_Layer_output=(self.batch_size,21,34,24)
        weights_upscore2=pickle.load(open("weights/upscore2_W.p","rb"))
        upscore2_stride=2
        self.upscore2_Layer=DeConvLayer(rng,upscore2_Layer_input,upscore2_Layer_input_shape,upscore2_Layer_filter,upscore2_Layer_output,upscore2_stride)
        self.upscore2_Layer.assignParams(weights_upscore2)



    def test(self,test_set_x):
        #out=self.relu_layer7_1.output
        #out=self.max_pool_layer3.output
        out=self.upscore2_Layer.output



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
        return np.array(outs)


def loadData():
    im = Image.open('/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code_Theano/fcn.berkeleyvision.org/data/pascal/VOCdevkit/VOC2012/JPEGImages/2007_000129.jpg')
    in_ = np.array(im, dtype=np.float64)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    return in_


if __name__=="__main__":
    im=loadData()
    print im.shape
    net=FCN(1,im.shape[1:])
    out=net.test(np.array([im]))
    #print out[0][10][0]
    print np.unravel_index(out.argmax(),out.shape)
    print out.shape
