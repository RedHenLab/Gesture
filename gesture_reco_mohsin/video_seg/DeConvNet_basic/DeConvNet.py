import numpy
import pylab
from PIL import Image
import pickle
import time
import matplotlib.pyplot as plt

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
        rng = numpy.random.RandomState(23488)

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



        # 28 x 28
        convLayer4_1_input=self.max_pool_layer3.output
        convLayer4_1_input_shape=(self.batch_size,num_features[6],28,28)
        convLayer4_1_filter=(num_features[7],num_features[6],3,3)
        weights_conv4_1=pickle.load(open("weights/conv4_1W.p","rb"))
        bias_conv4_1=pickle.load(open("weights/bias4_1.p","rb"))
        self.convLayer4_1=PaddedConvLayer(rng,convLayer4_1_input,convLayer4_1_input_shape,convLayer4_1_filter)
        self.convLayer4_1.assignParams(weights_conv4_1,bias_conv4_1)

        batch_norm_layer4_1_input=self.convLayer4_1.output
        bn4_1gamma=pickle.load(open("weights/bn4_1gamma.p"))
        bn4_1beta=pickle.load(open("weights/bn4_1beta.p"))
        self.batch_norm_layer4_1=CNNBatchNormLayer(batch_norm_layer4_1_input,num_features[7])
        self.batch_norm_layer4_1.assignParams(bn4_1gamma,bn4_1beta)

        relu_layer4_1_input=self.batch_norm_layer4_1.output
        self.relu_layer4_1=ReLuLayer(relu_layer4_1_input)


        convLayer4_2_input=self.relu_layer4_1.output
        convLayer4_2_input_shape=(self.batch_size,num_features[7],28,28)
        convLayer4_2_filter=(num_features[8],num_features[7],3,3)
        weights_conv4_2=pickle.load(open("weights/conv4_2W.p","rb"))
        bias_conv4_2=pickle.load(open("weights/bias4_2.p","rb"))
        self.convLayer4_2=PaddedConvLayer(rng,convLayer4_2_input,convLayer4_2_input_shape,convLayer4_2_filter)
        self.convLayer4_2.assignParams(weights_conv4_2,bias_conv4_2)

        batch_norm_layer4_2_input=self.convLayer4_2.output
        bn4_2gamma=pickle.load(open("weights/bn4_2gamma.p"))
        bn4_2beta=pickle.load(open("weights/bn4_2beta.p"))
        self.batch_norm_layer4_2=CNNBatchNormLayer(batch_norm_layer4_2_input,num_features[8])
        self.batch_norm_layer4_2.assignParams(bn4_2gamma,bn4_2beta)

        relu_layer4_2_input=self.batch_norm_layer4_2.output
        self.relu_layer4_2=ReLuLayer(relu_layer4_2_input)


        convLayer4_3_input=self.relu_layer4_2.output
        convLayer4_3_input_shape=(self.batch_size,num_features[8],28,28)
        convLayer4_3_filter=(num_features[9],num_features[8],3,3)
        weights_conv4_3=pickle.load(open("weights/conv4_3W.p","rb"))
        bias_conv4_3=pickle.load(open("weights/bias4_3.p","rb"))
        self.convLayer4_3=PaddedConvLayer(rng,convLayer4_3_input,convLayer4_3_input_shape,convLayer4_3_filter)
        self.convLayer4_3.assignParams(weights_conv4_3,bias_conv4_3)

        batch_norm_layer4_3_input=self.convLayer4_3.output
        bn4_3gamma=pickle.load(open("weights/bn4_3gamma.p"))
        bn4_3beta=pickle.load(open("weights/bn4_3beta.p"))
        self.batch_norm_layer4_3=CNNBatchNormLayer(batch_norm_layer4_3_input,num_features[9])
        self.batch_norm_layer4_3.assignParams(bn4_3gamma,bn4_3beta)

        relu_layer4_3_input=self.batch_norm_layer4_3.output
        self.relu_layer4_3=ReLuLayer(relu_layer4_3_input)

        max_pool_layer4_input=self.relu_layer4_3.output
        self.max_pool_layer4=SwitchedMaxPoolLayer(max_pool_layer4_input)



        # 14 x 14
        convLayer5_1_input=self.max_pool_layer4.output
        convLayer5_1_input_shape=(self.batch_size,num_features[9],14,14)
        convLayer5_1_filter=(num_features[10],num_features[9],3,3)
        weights_conv5_1=pickle.load(open("weights/conv5_1W.p","rb"))
        bias_conv5_1=pickle.load(open("weights/bias5_1.p","rb"))
        self.convLayer5_1=PaddedConvLayer(rng,convLayer5_1_input,convLayer5_1_input_shape,convLayer5_1_filter)
        self.convLayer5_1.assignParams(weights_conv5_1,bias_conv5_1)

        batch_norm_layer5_1_input=self.convLayer5_1.output
        bn5_1gamma=pickle.load(open("weights/bn5_1gamma.p"))
        bn5_1beta=pickle.load(open("weights/bn5_1beta.p"))
        self.batch_norm_layer5_1=CNNBatchNormLayer(batch_norm_layer5_1_input,num_features[10])
        self.batch_norm_layer5_1.assignParams(bn5_1gamma,bn5_1beta)

        relu_layer5_1_input=self.batch_norm_layer5_1.output
        self.relu_layer5_1=ReLuLayer(relu_layer5_1_input)


        convLayer5_2_input=self.relu_layer5_1.output
        convLayer5_2_input_shape=(self.batch_size,num_features[10],14,14)
        convLayer5_2_filter=(num_features[11],num_features[10],3,3)
        weights_conv5_2=pickle.load(open("weights/conv5_2W.p","rb"))
        bias_conv5_2=pickle.load(open("weights/bias5_2.p","rb"))
        self.convLayer5_2=PaddedConvLayer(rng,convLayer5_2_input,convLayer5_2_input_shape,convLayer5_2_filter)
        self.convLayer5_2.assignParams(weights_conv5_2,bias_conv5_2)

        batch_norm_layer5_2_input=self.convLayer5_2.output
        bn5_2gamma=pickle.load(open("weights/bn5_2gamma.p"))
        bn5_2beta=pickle.load(open("weights/bn5_2beta.p"))
        self.batch_norm_layer5_2=CNNBatchNormLayer(batch_norm_layer5_2_input,num_features[11])
        self.batch_norm_layer5_2.assignParams(bn5_2gamma,bn5_2beta)

        relu_layer5_2_input=self.batch_norm_layer5_2.output
        self.relu_layer5_2=ReLuLayer(relu_layer5_2_input)


        convLayer5_3_input=self.relu_layer5_2.output
        convLayer5_3_input_shape=(self.batch_size,num_features[11],14,14)
        convLayer5_3_filter=(num_features[12],num_features[11],3,3)
        weights_conv5_3=pickle.load(open("weights/conv5_3W.p","rb"))
        bias_conv5_3=pickle.load(open("weights/bias5_3.p","rb"))
        self.convLayer5_3=PaddedConvLayer(rng,convLayer5_3_input,convLayer5_3_input_shape,convLayer5_3_filter)
        self.convLayer5_3.assignParams(weights_conv5_3,bias_conv5_3)

        batch_norm_layer5_3_input=self.convLayer5_3.output
        bn5_3gamma=pickle.load(open("weights/bn5_3gamma.p"))
        bn5_3beta=pickle.load(open("weights/bn5_3beta.p"))
        self.batch_norm_layer5_3=CNNBatchNormLayer(batch_norm_layer5_3_input,num_features[12])
        self.batch_norm_layer5_3.assignParams(bn5_3gamma,bn5_3beta)

        relu_layer5_3_input=self.batch_norm_layer5_3.output
        self.relu_layer5_3=ReLuLayer(relu_layer5_3_input)

        max_pool_layer5_input=self.relu_layer5_3.output
        self.max_pool_layer5=SwitchedMaxPoolLayer(max_pool_layer5_input)



        # 7 x 7
        convLayer6_1_input=self.max_pool_layer5.output
        convLayer6_1_input_shape=(self.batch_size,num_features[12],7,7)
        convLayer6_1_filter=(num_features[13],num_features[12],7,7)
        weights_conv6_1=pickle.load(open("weights/conv6_1W.p","rb"))
        #print weights_conv6_1.shape
        bias_conv6_1=pickle.load(open("weights/bias6_1.p","rb"))
        self.convLayer6_1=ConvLayer(rng,convLayer6_1_input,convLayer6_1_input_shape,convLayer6_1_filter)
        self.convLayer6_1.assignParams(weights_conv6_1,bias_conv6_1)

        batch_norm_layer6_1_input=self.convLayer6_1.output
        bn6_1gamma=pickle.load(open("weights/bn6_1gamma.p"))
        bn6_1beta=pickle.load(open("weights/bn6_1beta.p"))
        self.batch_norm_layer6_1=CNNBatchNormLayer(batch_norm_layer6_1_input,num_features[13])
        self.batch_norm_layer6_1.assignParams(bn6_1gamma,bn6_1beta)

        relu_layer6_1_input=self.batch_norm_layer6_1.output
        self.relu_layer6_1=ReLuLayer(relu_layer6_1_input)



        # 1 x 1
        convLayer7_1_input=self.relu_layer6_1.output
        convLayer7_1_input_shape=(self.batch_size,num_features[13],7,7)
        convLayer7_1_filter=(num_features[14],num_features[13],7,7)
        weights_conv7_1=pickle.load(open("weights/conv7_1W.p","rb"))
        bias_conv7_1=pickle.load(open("weights/bias7_1.p","rb"))
        self.convLayer7_1=ConvLayer(rng,convLayer7_1_input,convLayer7_1_input_shape,convLayer7_1_filter)
        self.convLayer7_1.assignParams(weights_conv7_1,bias_conv7_1)

        batch_norm_layer7_1_input=self.convLayer7_1.output
        bn7_1gamma=pickle.load(open("weights/bn7_1gamma.p"))
        bn7_1beta=pickle.load(open("weights/bn7_1beta.p"))
        self.batch_norm_layer7_1=CNNBatchNormLayer(batch_norm_layer7_1_input,num_features[14])
        self.batch_norm_layer7_1.assignParams(bn7_1gamma,bn7_1beta)

        relu_layer7_1_input=self.batch_norm_layer7_1.output
        self.relu_layer7_1=ReLuLayer(relu_layer7_1_input)



        # 7 x 7
        deconvLayer6_1_input=self.relu_layer7_1.output
        deconvLayer6_1_input_shape=(self.batch_size,num_features[13],1,1)
        deconvLayer6_1_output_shape=(self.batch_size,num_features[12],7,7)
        deconvLayer6_1_filter=(num_features[13],num_features[12],7,7)
        weights_deconv6_1=pickle.load(open("weights/deconv6_1W.p","rb"))
        bias_deconv6_1=pickle.load(open("weights/deconvbias6_1.p","rb"))
        self.deconvLayer6_1=DeConvLayer(rng,deconvLayer6_1_input,deconvLayer6_1_input_shape,deconvLayer6_1_filter
        ,deconvLayer6_1_output_shape)
        self.deconvLayer6_1.assignParams(weights_deconv6_1,bias_deconv6_1)

        deconvbatch_norm_layer6_1_input=self.deconvLayer6_1.output
        deconvbn6_1gamma=pickle.load(open("weights/deconvbn6_1gamma.p"))
        deconvbn6_1beta=pickle.load(open("weights/deconvbn6_1beta.p"))
        self.deconvbatch_norm_layer6_1=CNNBatchNormLayer(deconvbatch_norm_layer6_1_input,num_features[12])
        self.deconvbatch_norm_layer6_1.assignParams(deconvbn6_1gamma,deconvbn6_1beta)

        deconvrelu_layer6_1_input=self.deconvbatch_norm_layer6_1.output
        self.deconvrelu_layer6_1=ReLuLayer(deconvrelu_layer6_1_input)

        unpool_layer5_input=self.deconvrelu_layer6_1.output
        unpool_layer5_switch=self.max_pool_layer5.switch
        self.unpool_layer5=UnPoolLayer(unpool_layer5_input,unpool_layer5_switch)



        # 14 x 14
        deconvLayer5_1_input=self.unpool_layer5.output
        deconvLayer5_1_input_shape=(self.batch_size,num_features[12],14,14)
        deconvLayer5_1_output_shape=(self.batch_size,num_features[11],14,14)
        deconvLayer5_1_filter=(num_features[12],num_features[11],3,3)
        weights_deconv5_1=pickle.load(open("weights/deconv5_1W.p","rb"))
        bias_deconv5_1=pickle.load(open("weights/deconvbias5_1.p","rb"))
        self.deconvLayer5_1=PaddedDeConvLayer(rng,deconvLayer5_1_input,deconvLayer5_1_input_shape,deconvLayer5_1_filter
        ,deconvLayer5_1_output_shape)
        self.deconvLayer5_1.assignParams(weights_deconv5_1,bias_deconv5_1)

        deconvbatch_norm_layer5_1_input=self.deconvLayer5_1.output
        deconvbn5_1gamma=pickle.load(open("weights/deconvbn5_1gamma.p"))
        deconvbn5_1beta=pickle.load(open("weights/deconvbn5_1beta.p"))
        self.deconvbatch_norm_layer5_1=CNNBatchNormLayer(deconvbatch_norm_layer5_1_input,num_features[11])
        self.deconvbatch_norm_layer5_1.assignParams(deconvbn5_1gamma,deconvbn5_1beta)

        deconvrelu_layer5_1_input=self.deconvbatch_norm_layer5_1.output
        self.deconvrelu_layer5_1=ReLuLayer(deconvrelu_layer5_1_input)


        deconvLayer5_2_input=self.deconvrelu_layer5_1.output
        deconvLayer5_2_input_shape=(self.batch_size,num_features[11],14,14)
        deconvLayer5_2_output_shape=(self.batch_size,num_features[10],14,14)
        deconvLayer5_2_filter=(num_features[11],num_features[10],3,3)
        weights_deconv5_2=pickle.load(open("weights/deconv5_2W.p","rb"))
        bias_deconv5_2=pickle.load(open("weights/deconvbias5_2.p","rb"))
        self.deconvLayer5_2=PaddedDeConvLayer(rng,deconvLayer5_2_input,deconvLayer5_2_input_shape,deconvLayer5_2_filter
        ,deconvLayer5_2_output_shape)
        self.deconvLayer5_2.assignParams(weights_deconv5_2,bias_deconv5_2)

        deconvbatch_norm_layer5_2_input=self.deconvLayer5_2.output
        deconvbn5_2gamma=pickle.load(open("weights/deconvbn5_2gamma.p"))
        deconvbn5_2beta=pickle.load(open("weights/deconvbn5_2beta.p"))
        self.deconvbatch_norm_layer5_2=CNNBatchNormLayer(deconvbatch_norm_layer5_2_input,num_features[10])
        self.deconvbatch_norm_layer5_2.assignParams(deconvbn5_2gamma,deconvbn5_2beta)

        deconvrelu_layer5_2_input=self.deconvbatch_norm_layer5_2.output
        self.deconvrelu_layer5_2=ReLuLayer(deconvrelu_layer5_2_input)


        deconvLayer5_3_input=self.deconvrelu_layer5_2.output
        deconvLayer5_3_input_shape=(self.batch_size,num_features[10],14,14)
        deconvLayer5_3_output_shape=(self.batch_size,num_features[9],14,14)
        deconvLayer5_3_filter=(num_features[10],num_features[9],3,3)
        weights_deconv5_3=pickle.load(open("weights/deconv5_3W.p","rb"))
        bias_deconv5_3=pickle.load(open("weights/deconvbias5_3.p","rb"))
        self.deconvLayer5_3=PaddedDeConvLayer(rng,deconvLayer5_3_input,deconvLayer5_3_input_shape,deconvLayer5_3_filter
        ,deconvLayer5_3_output_shape)
        self.deconvLayer5_3.assignParams(weights_deconv5_3,bias_deconv5_3)

        deconvbatch_norm_layer5_3_input=self.deconvLayer5_3.output
        deconvbn5_3gamma=pickle.load(open("weights/deconvbn5_3gamma.p"))
        deconvbn5_3beta=pickle.load(open("weights/deconvbn5_3beta.p"))
        self.deconvbatch_norm_layer5_3=CNNBatchNormLayer(deconvbatch_norm_layer5_3_input,num_features[9])
        self.deconvbatch_norm_layer5_3.assignParams(deconvbn5_3gamma,deconvbn5_3beta)

        deconvrelu_layer5_3_input=self.deconvbatch_norm_layer5_3.output
        self.deconvrelu_layer5_3=ReLuLayer(deconvrelu_layer5_3_input)


        unpool_layer4_input=self.deconvrelu_layer5_3.output
        unpool_layer4_switch=self.max_pool_layer4.switch
        self.unpool_layer4=UnPoolLayer(unpool_layer4_input,unpool_layer4_switch)



        # 28 x 28
        deconvLayer4_1_input=self.unpool_layer4.output
        deconvLayer4_1_input_shape=(self.batch_size,num_features[9],28,28)
        deconvLayer4_1_output_shape=(self.batch_size,num_features[8],28,28)
        deconvLayer4_1_filter=(num_features[9],num_features[8],3,3)
        weights_deconv4_1=pickle.load(open("weights/deconv4_1W.p","rb"))
        bias_deconv4_1=pickle.load(open("weights/deconvbias4_1.p","rb"))
        self.deconvLayer4_1=PaddedDeConvLayer(rng,deconvLayer4_1_input,deconvLayer4_1_input_shape,deconvLayer4_1_filter
        ,deconvLayer4_1_output_shape)
        self.deconvLayer4_1.assignParams(weights_deconv4_1,bias_deconv4_1)

        deconvbatch_norm_layer4_1_input=self.deconvLayer4_1.output
        deconvbn4_1gamma=pickle.load(open("weights/deconvbn4_1gamma.p"))
        deconvbn4_1beta=pickle.load(open("weights/deconvbn4_1beta.p"))
        self.deconvbatch_norm_layer4_1=CNNBatchNormLayer(deconvbatch_norm_layer4_1_input,num_features[8])
        self.deconvbatch_norm_layer4_1.assignParams(deconvbn4_1gamma,deconvbn4_1beta)

        deconvrelu_layer4_1_input=self.deconvbatch_norm_layer4_1.output
        self.deconvrelu_layer4_1=ReLuLayer(deconvrelu_layer4_1_input)


        deconvLayer4_2_input=self.deconvrelu_layer4_1.output
        deconvLayer4_2_input_shape=(self.batch_size,num_features[8],28,28)
        deconvLayer4_2_output_shape=(self.batch_size,num_features[7],28,28)
        deconvLayer4_2_filter=(num_features[8],num_features[7],3,3)
        weights_deconv4_2=pickle.load(open("weights/deconv4_2W.p","rb"))
        bias_deconv4_2=pickle.load(open("weights/deconvbias4_2.p","rb"))
        self.deconvLayer4_2=PaddedDeConvLayer(rng,deconvLayer4_2_input,deconvLayer4_2_input_shape,deconvLayer4_2_filter
        ,deconvLayer4_2_output_shape)
        self.deconvLayer4_2.assignParams(weights_deconv4_2,bias_deconv4_2)

        deconvbatch_norm_layer4_2_input=self.deconvLayer4_2.output
        deconvbn4_2gamma=pickle.load(open("weights/deconvbn4_2gamma.p"))
        deconvbn4_2beta=pickle.load(open("weights/deconvbn4_2beta.p"))
        self.deconvbatch_norm_layer4_2=CNNBatchNormLayer(deconvbatch_norm_layer4_2_input,num_features[7])
        self.deconvbatch_norm_layer4_2.assignParams(deconvbn4_2gamma,deconvbn4_2beta)

        deconvrelu_layer4_2_input=self.deconvbatch_norm_layer4_2.output
        self.deconvrelu_layer4_2=ReLuLayer(deconvrelu_layer4_2_input)


        deconvLayer4_3_input=self.deconvrelu_layer4_2.output
        deconvLayer4_3_input_shape=(self.batch_size,num_features[7],28,28)
        deconvLayer4_3_output_shape=(self.batch_size,num_features[6],28,28)
        deconvLayer4_3_filter=(num_features[7],num_features[6],3,3)
        weights_deconv4_3=pickle.load(open("weights/deconv4_3W.p","rb"))
        bias_deconv4_3=pickle.load(open("weights/deconvbias4_3.p","rb"))
        self.deconvLayer4_3=PaddedDeConvLayer(rng,deconvLayer4_3_input,deconvLayer4_3_input_shape,deconvLayer4_3_filter
        ,deconvLayer4_3_output_shape)
        self.deconvLayer4_3.assignParams(weights_deconv4_3,bias_deconv4_3)

        deconvbatch_norm_layer4_3_input=self.deconvLayer4_3.output
        deconvbn4_3gamma=pickle.load(open("weights/deconvbn4_3gamma.p"))
        deconvbn4_3beta=pickle.load(open("weights/deconvbn4_3beta.p"))
        self.deconvbatch_norm_layer4_3=CNNBatchNormLayer(deconvbatch_norm_layer4_3_input,num_features[6])
        self.deconvbatch_norm_layer4_3.assignParams(deconvbn4_3gamma,deconvbn4_3beta)

        deconvrelu_layer4_3_input=self.deconvbatch_norm_layer4_3.output
        self.deconvrelu_layer4_3=ReLuLayer(deconvrelu_layer4_3_input)


        unpool_layer3_input=self.deconvrelu_layer4_3.output
        unpool_layer3_switch=self.max_pool_layer3.switch
        self.unpool_layer3=UnPoolLayer(unpool_layer3_input,unpool_layer3_switch)



        # 56 x 56
        deconvLayer3_1_input=self.unpool_layer3.output
        deconvLayer3_1_input_shape=(self.batch_size,num_features[6],56,56)
        deconvLayer3_1_output_shape=(self.batch_size,num_features[5],56,56)
        deconvLayer3_1_filter=(num_features[6],num_features[5],3,3)
        weights_deconv3_1=pickle.load(open("weights/deconv3_1W.p","rb"))
        bias_deconv3_1=pickle.load(open("weights/deconvbias3_1.p","rb"))
        self.deconvLayer3_1=PaddedDeConvLayer(rng,deconvLayer3_1_input,deconvLayer3_1_input_shape,deconvLayer3_1_filter
        ,deconvLayer3_1_output_shape)
        self.deconvLayer3_1.assignParams(weights_deconv3_1,bias_deconv3_1)

        deconvbatch_norm_layer3_1_input=self.deconvLayer3_1.output
        deconvbn3_1gamma=pickle.load(open("weights/deconvbn3_1gamma.p"))
        deconvbn3_1beta=pickle.load(open("weights/deconvbn3_1beta.p"))
        self.deconvbatch_norm_layer3_1=CNNBatchNormLayer(deconvbatch_norm_layer3_1_input,num_features[5])
        self.deconvbatch_norm_layer3_1.assignParams(deconvbn3_1gamma,deconvbn3_1beta)

        deconvrelu_layer3_1_input=self.deconvbatch_norm_layer3_1.output
        self.deconvrelu_layer3_1=ReLuLayer(deconvrelu_layer3_1_input)


        deconvLayer3_2_input=self.deconvrelu_layer3_1.output
        deconvLayer3_2_input_shape=(self.batch_size,num_features[5],56,56)
        deconvLayer3_2_output_shape=(self.batch_size,num_features[4],56,56)
        deconvLayer3_2_filter=(num_features[5],num_features[4],3,3)
        weights_deconv3_2=pickle.load(open("weights/deconv3_2W.p","rb"))
        bias_deconv3_2=pickle.load(open("weights/deconvbias3_2.p","rb"))
        self.deconvLayer3_2=PaddedDeConvLayer(rng,deconvLayer3_2_input,deconvLayer3_2_input_shape,deconvLayer3_2_filter
        ,deconvLayer3_2_output_shape)
        self.deconvLayer3_2.assignParams(weights_deconv3_2,bias_deconv3_2)

        deconvbatch_norm_layer3_2_input=self.deconvLayer3_2.output
        deconvbn3_2gamma=pickle.load(open("weights/deconvbn3_2gamma.p"))
        deconvbn3_2beta=pickle.load(open("weights/deconvbn3_2beta.p"))
        self.deconvbatch_norm_layer3_2=CNNBatchNormLayer(deconvbatch_norm_layer3_2_input,num_features[7])
        self.deconvbatch_norm_layer3_2.assignParams(deconvbn3_2gamma,deconvbn3_2beta)

        deconvrelu_layer3_2_input=self.deconvbatch_norm_layer3_2.output
        self.deconvrelu_layer3_2=ReLuLayer(deconvrelu_layer3_2_input)


        deconvLayer3_3_input=self.deconvrelu_layer3_2.output
        deconvLayer3_3_input_shape=(self.batch_size,num_features[4],56,56)
        deconvLayer3_3_output_shape=(self.batch_size,num_features[3],56,56)
        deconvLayer3_3_filter=(num_features[4],num_features[3],3,3)
        weights_deconv3_3=pickle.load(open("weights/deconv3_3W.p","rb"))
        bias_deconv3_3=pickle.load(open("weights/deconvbias3_3.p","rb"))
        self.deconvLayer3_3=PaddedDeConvLayer(rng,deconvLayer3_3_input,deconvLayer3_3_input_shape,deconvLayer3_3_filter
        ,deconvLayer3_3_output_shape)
        self.deconvLayer3_3.assignParams(weights_deconv3_3,bias_deconv3_3)

        deconvbatch_norm_layer3_3_input=self.deconvLayer3_3.output
        deconvbn3_3gamma=pickle.load(open("weights/deconvbn3_3gamma.p"))
        deconvbn3_3beta=pickle.load(open("weights/deconvbn3_3beta.p"))
        self.deconvbatch_norm_layer3_3=CNNBatchNormLayer(deconvbatch_norm_layer3_3_input,num_features[3])
        self.deconvbatch_norm_layer3_3.assignParams(deconvbn3_3gamma,deconvbn3_3beta)

        deconvrelu_layer3_3_input=self.deconvbatch_norm_layer3_3.output
        self.deconvrelu_layer3_3=ReLuLayer(deconvrelu_layer3_3_input)


        unpool_layer2_input=self.deconvrelu_layer3_3.output
        unpool_layer2_switch=self.max_pool_layer2.switch
        self.unpool_layer2=UnPoolLayer(unpool_layer2_input,unpool_layer2_switch)



        # 112 x 112
        deconvLayer2_1_input=self.unpool_layer2.output
        deconvLayer2_1_input_shape=(self.batch_size,num_features[3],112,112)
        deconvLayer2_1_output_shape=(self.batch_size,num_features[2],112,112)
        deconvLayer2_1_filter=(num_features[3],num_features[2],3,3)
        weights_deconv2_1=pickle.load(open("weights/deconv2_1W.p","rb"))
        bias_deconv2_1=pickle.load(open("weights/deconvbias2_1.p","rb"))
        self.deconvLayer2_1=PaddedDeConvLayer(rng,deconvLayer2_1_input,deconvLayer2_1_input_shape,deconvLayer2_1_filter
        ,deconvLayer2_1_output_shape)
        self.deconvLayer2_1.assignParams(weights_deconv2_1,bias_deconv2_1)

        deconvbatch_norm_layer2_1_input=self.deconvLayer2_1.output
        deconvbn2_1gamma=pickle.load(open("weights/deconvbn2_1gamma.p"))
        deconvbn2_1beta=pickle.load(open("weights/deconvbn2_1beta.p"))
        self.deconvbatch_norm_layer2_1=CNNBatchNormLayer(deconvbatch_norm_layer2_1_input,num_features[2])
        self.deconvbatch_norm_layer2_1.assignParams(deconvbn2_1gamma,deconvbn2_1beta)

        deconvrelu_layer2_1_input=self.deconvbatch_norm_layer2_1.output
        self.deconvrelu_layer2_1=ReLuLayer(deconvrelu_layer2_1_input)


        deconvLayer2_2_input=self.deconvrelu_layer2_1.output
        deconvLayer2_2_input_shape=(self.batch_size,num_features[2],112,112)
        deconvLayer2_2_output_shape=(self.batch_size,num_features[1],112,112)
        deconvLayer2_2_filter=(num_features[2],num_features[1],3,3)
        weights_deconv2_2=pickle.load(open("weights/deconv2_2W.p","rb"))
        bias_deconv2_2=pickle.load(open("weights/deconvbias2_2.p","rb"))
        self.deconvLayer2_2=PaddedDeConvLayer(rng,deconvLayer2_2_input,deconvLayer2_2_input_shape,deconvLayer2_2_filter
        ,deconvLayer2_2_output_shape)
        self.deconvLayer2_2.assignParams(weights_deconv2_2,bias_deconv2_2)

        deconvbatch_norm_layer2_2_input=self.deconvLayer2_2.output
        deconvbn2_2gamma=pickle.load(open("weights/deconvbn2_2gamma.p"))
        deconvbn2_2beta=pickle.load(open("weights/deconvbn2_2beta.p"))
        self.deconvbatch_norm_layer2_2=CNNBatchNormLayer(deconvbatch_norm_layer2_2_input,num_features[1])
        self.deconvbatch_norm_layer2_2.assignParams(deconvbn2_2gamma,deconvbn2_2beta)

        deconvrelu_layer2_2_input=self.deconvbatch_norm_layer2_2.output
        self.deconvrelu_layer2_2=ReLuLayer(deconvrelu_layer2_2_input)


        unpool_layer1_input=self.deconvrelu_layer2_2.output
        unpool_layer1_switch=self.max_pool_layer1.switch
        self.unpool_layer1=UnPoolLayer(unpool_layer1_input,unpool_layer1_switch)


        # 224 x 224
        deconvLayer1_1_input=self.unpool_layer1.output
        deconvLayer1_1_input_shape=(self.batch_size,num_features[0],224,224)
        deconvLayer1_1_output_shape=(self.batch_size,num_features[0],224,224)
        deconvLayer1_1_filter=(num_features[0],num_features[0],3,3)
        weights_deconv1_1=pickle.load(open("weights/deconv1_1W.p","rb"))
        bias_deconv1_1=pickle.load(open("weights/deconvbias1_1.p","rb"))
        self.deconvLayer1_1=PaddedDeConvLayer(rng,deconvLayer1_1_input,deconvLayer1_1_input_shape,deconvLayer1_1_filter
        ,deconvLayer1_1_output_shape)
        self.deconvLayer1_1.assignParams(weights_deconv1_1,bias_deconv1_1)

        deconvbatch_norm_layer1_1_input=self.deconvLayer1_1.output
        deconvbn1_1gamma=pickle.load(open("weights/deconvbn1_1gamma.p"))
        deconvbn1_1beta=pickle.load(open("weights/deconvbn1_1beta.p"))
        self.deconvbatch_norm_layer1_1=CNNBatchNormLayer(deconvbatch_norm_layer1_1_input,num_features[0])
        self.deconvbatch_norm_layer1_1.assignParams(deconvbn1_1gamma,deconvbn1_1beta)

        deconvrelu_layer1_1_input=self.deconvbatch_norm_layer1_1.output
        self.deconvrelu_layer1_1=ReLuLayer(deconvrelu_layer1_1_input)


        deconvLayer1_2_input=self.deconvrelu_layer1_1.output
        deconvLayer1_2_input_shape=(self.batch_size,num_features[0],224,224)
        deconvLayer1_2_output_shape=(self.batch_size,num_features[1],224,224)
        deconvLayer1_2_filter=(num_features[0],num_features[0],3,3)
        weights_deconv1_2=pickle.load(open("weights/deconv1_2W.p","rb"))
        bias_deconv1_2=pickle.load(open("weights/deconvbias1_2.p","rb"))
        self.deconvLayer1_2=PaddedDeConvLayer(rng,deconvLayer1_2_input,deconvLayer1_2_input_shape,deconvLayer1_2_filter
        ,deconvLayer1_2_output_shape)
        self.deconvLayer1_2.assignParams(weights_deconv1_2,bias_deconv1_2)

        deconvbatch_norm_layer1_2_input=self.deconvLayer1_2.output
        deconvbn1_2gamma=pickle.load(open("weights/deconvbn1_2gamma.p"))
        deconvbn1_2beta=pickle.load(open("weights/deconvbn1_2beta.p"))
        self.deconvbatch_norm_layer1_2=CNNBatchNormLayer(deconvbatch_norm_layer1_2_input,num_features[1])
        self.deconvbatch_norm_layer1_2.assignParams(deconvbn1_2gamma,deconvbn1_2beta)

        deconvrelu_layer1_2_input=self.deconvbatch_norm_layer1_2.output
        self.deconvrelu_layer1_2=ReLuLayer(deconvrelu_layer1_2_input)



        # output
        outLayer_input=self.deconvrelu_layer1_2.output
        outLayer_input_shape=(self.batch_size,num_features[0],224,224)
        outLayer_filter=(num_features[0],num_features[0],1,1)
        weights_out=pickle.load(open("weights/out_W.p","rb"))
        bias_out=pickle.load(open("weights/bias_out.p","rb"))
        self.outLayer=ConvLayer(rng,outLayer_input,outLayer_input_shape,outLayer_filter)
        self.outLayer.assignParams(weights_out,bias_out)



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
        #out=self.unpool_layer2.output

        #out=self.relu_layer7_1.output

        #out=self.deconvLayer1_2.output
        out=self.outLayer.output

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



def processOutput(data):
    #out_shape=(data.shape[0])+data.shape[2:]
    #outs=np.zeros(out_shape)
    outs=numpy.argmax(data,axis=1)
    print outs.shape
    return outs


def genImage(data):
    class_label=15
    print data
    img_shape=data.shape#+(3,)

    for class_label in range(20):

        img_vals=numpy.zeros(img_shape)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i][j]==class_label:
                    #print "hello"
                    img_vals[i][j]=255
                    #img_vals[i][j][1]=255
                    #img_vals[i][j][2]=255
        #print img_vals

        img=Image.fromarray(img_vals,'L')
        img.show(title="class"+str(class_label))


def genImagePlot(data):

    class_label=18

    plotY=[]
    plotX=[]

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j]==class_label:

                plotX.append(j)
                plotY.append(240-i)
    plt.plot(plotX,plotY,"ro")
    plt.show()



if __name__=="__main__":
    """
    deNet=DeConvNet(3,[64,64,128,128,256,256,256,512,512,512,512,512,512,4096,4096])
    numpy.set_printoptions(threshold='nan')
    print "loading data"
    data=loadData()
    print "finished loading data"

    start_time=time.time()
    outs=deNet.test(data)
    #print data[0]
    #print data[1][0][12][90:100]
    #print data[1][0][13][90:100]

    #print "outs"
    #print len(outs)
    #print outs[1].shape
    #print outs[1][0][12][90:100]
    #print outs[1][0][13][90:100]
    #print outs[1][0]

    new_out=processOutput(outs)
    pickle.dump(new_out,open("outs.p","wb"))
    #print new_out[0][0]

    print "elpased time ="+str(time.time()-start_time)
    """

    new_out=pickle.load(open("outs.p","rb"))
    numpy.set_printoptions(threshold='nan')
    print new_out[6]
    genImagePlot(new_out[6])
    #"""
