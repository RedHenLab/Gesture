import numpy
import pylab
from PIL import Image
import pickle
import time

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
        weights_conv1=pickle.load(open("weights/weights1.p","rb"))
        bias_conv1=pickle.load(open("weights/bias1.p","rb"))
        self.convLayer1=PaddedConvLayer(rng,convLayer1_input,convLayer1_input_shape,convLayer1_filter)
        self.convLayer1.assignParams(weights_conv1,bias_conv1)

        batch_norm_layer1_input=self.convLayer1.output
        #batch_norm_layer1_input=x.reshape((self.batch_size,3,224,224))
        bn1_1gamma=pickle.load(open("weights/bn1_1gamma.p"))
        bn1_1beta=pickle.load(open("weights/bn1_1beta.p"))
        self.batch_norm_layer1=CNNBatchNormLayer(batch_norm_layer1_input,num_features[0])
        #self.batch_norm_layer1=CNNBatchNormLayer(batch_norm_layer1_input,3)
        self.batch_norm_layer1.assignParams(bn1_1gamma,bn1_1beta)

        relu_layer1_1_input=self.batch_norm_layer1.output
        self.relu_layer1_1=ReLuLayer(relu_layer1_1_input)

        convLayer1_2_input=self.relu_layer1_1.output
        convLayer1_2_input_shape=(self.batch_size,num_features[0],224,224)
        convLayer1_2_filter=(num_features[1],num_features[0],3,3)
        weights_conv1_2=pickle.load(open("weights/conv1_2W.p","rb"))
        bias_conv1_2=pickle.load(open("weights/bias1_2.p","rb"))
        self.convLayer1_2=PaddedConvLayer(rng,convLayer1_2_input,convLayer1_2_input_shape,convLayer1_2_filter)
        self.convLayer1_2.assignParams(weights_conv1_2,bias_conv1_2)

        batch_norm_layer1_2_input=self.convLayer1_2.output
        bn1_2gamma=pickle.load(open("weights/bn1_2gamma.p"))
        bn1_2beta=pickle.load(open("weights/bn1_2beta.p"))
        self.batch_norm_layer1_2=CNNBatchNormLayer(batch_norm_layer1_2_input,num_features[1])
        self.batch_norm_layer1_2.assignParams(bn1_2gamma,bn1_2beta)

        relu_layer1_2_input=self.batch_norm_layer1_2.output
        self.relu_layer1_2=ReLuLayer(relu_layer1_2_input)

        #max_pool_layer1_input=x.reshape((self.batch_size,3,224,224))
        max_pool_layer1_input=self.relu_layer1_2.output
        self.max_pool_layer1=SwitchedMaxPoolLayer(max_pool_layer1_input)



        # 112 x 112
        convLayer2_1_input=self.max_pool_layer1.output
        convLayer2_1_input_shape=(self.batch_size,num_features[1],112,112)
        convLayer2_1_filter=(num_features[2],num_features[1],3,3)
        weights_conv2_1=pickle.load(open("weights/conv2_1W.p","rb"))
        bias_conv2_1=pickle.load(open("weights/bias2_1.p","rb"))
        self.convLayer2_1=PaddedConvLayer(rng,convLayer2_1_input,convLayer2_1_input_shape,convLayer2_1_filter)
        self.convLayer2_1.assignParams(weights_conv2_1,bias_conv2_1)

        batch_norm_layer2_1_input=self.convLayer2_1.output
        bn2_1gamma=pickle.load(open("weights/bn2_1gamma.p"))
        bn2_1beta=pickle.load(open("weights/bn2_1beta.p"))
        self.batch_norm_layer2_1=CNNBatchNormLayer(batch_norm_layer2_1_input,num_features[2])
        self.batch_norm_layer2_1.assignParams(bn2_1gamma,bn2_1beta)

        relu_layer2_1_input=self.batch_norm_layer2_1.output
        self.relu_layer2_1=ReLuLayer(relu_layer2_1_input)


        convLayer2_2_input=self.relu_layer2_1.output
        convLayer2_2_input_shape=(self.batch_size,num_features[2],112,112)
        convLayer2_2_filter=(num_features[3],num_features[2],3,3)
        weights_conv2_2=pickle.load(open("weights/conv2_2W.p","rb"))
        bias_conv2_2=pickle.load(open("weights/bias2_2.p","rb"))
        self.convLayer2_2=PaddedConvLayer(rng,convLayer2_2_input,convLayer2_2_input_shape,convLayer2_2_filter)
        self.convLayer2_2.assignParams(weights_conv2_2,bias_conv2_2)

        batch_norm_layer2_2_input=self.convLayer2_2.output
        bn2_2gamma=pickle.load(open("weights/bn2_2gamma.p"))
        bn2_2beta=pickle.load(open("weights/bn2_2beta.p"))
        self.batch_norm_layer2_2=CNNBatchNormLayer(batch_norm_layer2_2_input,num_features[3])
        self.batch_norm_layer2_2.assignParams(bn2_2gamma,bn2_2beta)

        relu_layer2_2_input=self.batch_norm_layer2_2.output
        self.relu_layer2_2=ReLuLayer(relu_layer2_2_input)

        max_pool_layer2_input=self.relu_layer2_2.output
        self.max_pool_layer2=SwitchedMaxPoolLayer(max_pool_layer2_input)



        # 56 x 56
        convLayer3_1_input=self.max_pool_layer2.output
        convLayer3_1_input_shape=(self.batch_size,num_features[3],56,56)
        convLayer3_1_filter=(num_features[4],num_features[3],3,3)
        weights_conv3_1=pickle.load(open("weights/conv3_1W.p","rb"))
        bias_conv3_1=pickle.load(open("weights/bias3_1.p","rb"))
        self.convLayer3_1=PaddedConvLayer(rng,convLayer3_1_input,convLayer3_1_input_shape,convLayer3_1_filter)
        self.convLayer3_1.assignParams(weights_conv3_1,bias_conv3_1)

        batch_norm_layer3_1_input=self.convLayer3_1.output
        bn3_1gamma=pickle.load(open("weights/bn3_1gamma.p"))
        bn3_1beta=pickle.load(open("weights/bn3_1beta.p"))
        self.batch_norm_layer3_1=CNNBatchNormLayer(batch_norm_layer3_1_input,num_features[4])
        self.batch_norm_layer3_1.assignParams(bn3_1gamma,bn3_1beta)

        relu_layer3_1_input=self.batch_norm_layer3_1.output
        self.relu_layer3_1=ReLuLayer(relu_layer3_1_input)


        convLayer3_2_input=self.relu_layer3_1.output
        convLayer3_2_input_shape=(self.batch_size,num_features[4],56,56)
        convLayer3_2_filter=(num_features[5],num_features[4],3,3)
        weights_conv3_2=pickle.load(open("weights/conv3_2W.p","rb"))
        bias_conv3_2=pickle.load(open("weights/bias3_2.p","rb"))
        self.convLayer3_2=PaddedConvLayer(rng,convLayer3_2_input,convLayer3_2_input_shape,convLayer3_2_filter)
        self.convLayer3_2.assignParams(weights_conv3_2,bias_conv3_2)

        batch_norm_layer3_2_input=self.convLayer3_2.output
        bn3_2gamma=pickle.load(open("weights/bn3_2gamma.p"))
        bn3_2beta=pickle.load(open("weights/bn3_2beta.p"))
        self.batch_norm_layer3_2=CNNBatchNormLayer(batch_norm_layer3_2_input,num_features[5])
        self.batch_norm_layer3_2.assignParams(bn3_2gamma,bn3_2beta)

        relu_layer3_2_input=self.batch_norm_layer3_2.output
        self.relu_layer3_2=ReLuLayer(relu_layer3_2_input)


        convLayer3_3_input=self.relu_layer3_2.output
        convLayer3_3_input_shape=(self.batch_size,num_features[5],56,56)
        convLayer3_3_filter=(num_features[6],num_features[5],3,3)
        weights_conv3_3=pickle.load(open("weights/conv3_3W.p","rb"))
        bias_conv3_3=pickle.load(open("weights/bias3_3.p","rb"))
        self.convLayer3_3=PaddedConvLayer(rng,convLayer3_3_input,convLayer3_3_input_shape,convLayer3_3_filter)
        self.convLayer3_3.assignParams(weights_conv3_3,bias_conv3_3)

        batch_norm_layer3_3_input=self.convLayer3_3.output
        bn3_3gamma=pickle.load(open("weights/bn3_3gamma.p"))
        bn3_3beta=pickle.load(open("weights/bn3_3beta.p"))
        self.batch_norm_layer3_3=CNNBatchNormLayer(batch_norm_layer3_3_input,num_features[6])
        self.batch_norm_layer3_3.assignParams(bn3_3gamma,bn3_3beta)

        relu_layer3_3_input=self.batch_norm_layer3_3.output
        self.relu_layer3_3=ReLuLayer(relu_layer3_3_input)

        max_pool_layer3_input=self.relu_layer3_3.output
        self.max_pool_layer3=SwitchedMaxPoolLayer(max_pool_layer3_input)


        unpool_layer1_input=self.max_pool_layer3.output
        unpool_layer1_switch=self.max_pool_layer3.switch
        self.unpool_layer1=UnPoolLayer(unpool_layer1_input,unpool_layer1_switch)


    def test(self,test_set_x):
        #out=self.batch_norm_layer2_2.output

        # Code for testing batch convolution
        #out_mean=self.batch_norm_layer1.batch_mean
        #out_var=self.batch_norm_layer1.batch_var
        #out_gamma=self.batch_norm_layer1.gamma

        # Code for testing swtiched max pooling
        #switch=self.max_pool_layer1.switch
        #out=self.max_pool_layer2.output
        #out=self.max_pool_layer2.switch

        # Code for testing unpooling layer
        out=self.unpool_layer1.output

        #out=self.relu_layer2_2.output

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
    deNet=DeConvNet(3,[64,64,128,128,256,256,256,512,512,512])
    numpy.set_printoptions(threshold='nan')
    print "loading data"
    data=loadData()
    print "finished loading data"

    start_time=time.time()
    outs=deNet.test(data)
    #print data[0]
    print data[1][0][12][90:100]
    print data[1][0][13][90:100]

    print "outs"
    print len(outs)
    print outs[1].shape
    print outs[1][0][12][10:20]
    print outs[1][0][13][10:20]

    print "elpased time ="+str(time.time()-start_time)
