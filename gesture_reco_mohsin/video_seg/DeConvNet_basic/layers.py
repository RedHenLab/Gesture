import numpy
import pylab
from PIL import Image
import pickle

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from raw_pool_theano import *

import lasagne

"""
The output image size of the convolution layer is
the same as the input.
"""
class PaddedConvLayer(object):

    def __init__(self,rng,inputData,image_shape,filter_shape):

        """
        rng : Random number generator
        input : tensor4(batch_size,num_input_feature_maps,height,width)
        filter_shape:tensor4(num_features,num_input_feature_maps,filter height, filter_width)
        out_features_shape=tensor4(batch_size,num)
        """

        assert image_shape[1]==filter_shape[1]

        self.input=inputData

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) )

        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))

        self.W=theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound,high=W_bound,size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # adding a padding to get the same size of output and input.
        #padding=(filter_shape[2]-1)/2
        conv_out=conv2d(self.input,self.W,border_mode='half')

        self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params=[self.W,self.b]


    def assignParams(self,W,b):
        updates_W=(self.W,W)
        updates_b=(self.b,b)

        assignW=theano.function(
            inputs=[],
            updates=updates_W
        )

        assignb=theano.function(
            inputs=[],
            updates=updates_b
        )

        assignW()
        assignb()

"""
Batch normalization layer. The output and the input are
of the same size. It is just normalized across batch and
shifted to maintain the effect of non linearity
"""
class CNNBatchNormLayer(object):

    def __init__(self,inputData,num_out):
        self.input=inputData

        gamma_values = numpy.ones((num_out,), dtype=theano.config.floatX)
        self.gamma = theano.shared(value=gamma_values, borrow=True)

        beta_values = numpy.zeros((num_out,), dtype=theano.config.floatX)
        self.beta = theano.shared(value=beta_values, borrow=True)

        batch_mean=T.mean(self.input,keepdims=True,axis=0)
        batch_var=T.var(self.input,keepdims=True,axis=0)

        self.batch_mean=batch_mean
        self.batch_var=T.pow(batch_var,0.5)

        batch_normalize=(inputData-batch_mean)/(T.pow(batch_var,0.5))

        self.beta = self.beta.dimshuffle('x', 0, 'x', 'x')
        self.gamma = self.gamma.dimshuffle('x', 0, 'x', 'x')

        self.output=batch_normalize*self.gamma+self.beta

        self.params=[self.gamma,self.beta]


    def assignParams(self,gamma,beta):
        updates_gamma=(self.gamma,gamma)
        updates_beta=(self.beta,beta)

        assignGamma=theano.function(
            inputs=[],
            updates=updates_gamma
        )

        assignBeta=theano.function(
            inputs=[],
            updates=updates_beta
        )

        assignGamma()
        assignBeta()



class ReLuLayer(object):

    def __init__(self,inputData):
        self.input=inputData
        self.output=T.nnet.relu(self.input)



class MaxPoolLayer(object):

    def __init__(self,inputData,poolsize=(2,2)):
        self.input=inputData

        pooled_out=downsample.max_pool_2d(
            input=self.input,
            ds=poolsize,
            ignore_border=True
        )

        self.output=pooled_out


class SwitchedMaxPoolLayer(object):

    def __init__(self,inputData,poolsize=(2,2)):
        self.input=inputData

        switch_out=pool_2d(
            input=self.input,
            ds=poolsize,
            ignore_border=True
        )

        self.switch=switch_out

        pooled_out=downsample.max_pool_2d(
            input=self.input,
            ds=poolsize,
            ignore_border=True
        )

        self.output=pooled_out


class PaddedDeConvLayer(object):

    def __init__(self,rng,inputData,image_shape,filter_shape):
        self.input=inputData

        self.deConvLayer=lasagne.layers.TransposedConv2DLayer(self.input,num_filters=filter_shape[0],
        filter_size=(filter_shape[2],filter_shape[3]),nonlinearity=lasagne.nonlinearities.linear,
        pad='same')

        self.params=self.deConvLayer.get_params()

        self.output=self.deConvLayer.get_output_for(self.input)


    def assignParams(self,W,b):
        updates_W=(self.params[0],W)
        updates_b=(self.params[1],b)

        assignW=theano.function(
            inputs=[],
            updates=updates_W
        )

        assignb=theano.function(
            inputs=[],
            updates=updates_b
        )

        assignW()
        assignb()


class UnPoolLayer(object):

    def __init__(self,inputData,switchedData,poolsize=(2,2)):
        self.input=inputData
        self.switch=switchedData

        output=unpool_2d(
            input=self.input,
            ds=poolsize,
            switch=self.switch,
            ignore_border=True
        )

        self.output=output
