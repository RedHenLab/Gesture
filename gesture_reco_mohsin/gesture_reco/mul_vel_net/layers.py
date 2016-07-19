import numpy
import pylab
from PIL import Image
import pickle

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
#from theano.tensor.nnet.conv3d2d import conv3d
from theano.tensor.signal import downsample
from raw_pool_theano import *
from raw_theano_conv3d import *
from raw_theano import *

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
        conv_out=conv2d(self.input,self.W,border_mode='half',filter_flip=False)

        #self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        self.params=[self.W,self.b]


    def assignParams(self,W,b):
        updates_W=(self.W,W)
        updates_b=(self.b,b)

        assignW=theano.function(
            inputs=[],
            updates=[updates_W]
        )

        assignb=theano.function(
            inputs=[],
            updates=[updates_b]
        )

        assignW()
        assignb()



class TemporalConvLayer(object):

    def __init__(self,rng,inputData,image_shape,filter_shape,temporal_stride=1,filter_stride=1):
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

        conv_out=conv3d(self.input,self.W)
        conv_out=conv_out[:,0::temporal_stride,:,0::filter_stride,0::filter_stride]

        self.output=conv_out

        self.params=[self.W]


    def assignParams(self,W,b):
        updates_W=(self.W,W)
        updates_b=(self.b,b)

        assignW=theano.function(
            inputs=[],
            updates=[updates_W]
        )

        assignb=theano.function(
            inputs=[],
            updates=[updates_b]
        )

        assignW()
        assignb()



class TemporalDeConvLayer(object):
    """
    The filter shape is:
    (num_input_layers,temporal size,num_output_channels,height,width)


    """

    def __init__(self,rng,inputData,out_shape,filter_shape,temporal_stride=1,filter_stride=1):
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

        self.conv_input=theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound,high=W_bound,size=out_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        self.conv_input=theano.shared(
            numpy.zeros(out_shape,dtype=theano.config.floatX),
            borrow=True
        )

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out=conv3d(self.conv_input,self.W)
        conv_out=conv_out[:,0::temporal_stride,:,0::filter_stride,0::filter_stride]
        conv_out=T.mul(conv_out,self.input)

        #assignVals(conv_out,self.input)
        back_stride=T.grad(T.sum(conv_out),self.conv_input)

        #update_conv=(conv_out,self.input)
        #assignUpdate=theano.function(
        #    inputs=[],
        #    updates=[update_conv]
        #)

        #out_func=theano.function(
        #    inputs=[],
        #    outputs=[back_stride],
            #givens={conv_out: self.input.eval()}
        #)

        #assignUpdate()
        #self.output=out_func()[0]
        #self.output=self.input.eval()
        self.output=back_stride

        self.params=[self.W]


    def assignParams(self,W,b):
        updates_W=(self.W,W)
        updates_b=(self.b,b)

        assignW=theano.function(
            inputs=[],
            updates=[updates_W]
        )

        assignb=theano.function(
            inputs=[],
            updates=[updates_b]
        )

        assignW()
        assignb()



class ConvLayer(object):

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
        conv_out=conv2d(self.input,self.W,filter_flip=False)

        #self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        self.params=[self.W,self.b]


    def assignParams(self,W,b):
        updates_W=(self.W,W)
        updates_b=(self.b,b)

        assignW=theano.function(
            inputs=[],
            updates=[updates_W]
        )

        assignb=theano.function(
            inputs=[],
            updates=[updates_b]
        )

        assignW()
        assignb()


"""
Batch normalization layer. The output and the input are
of the same size. It is just normalized across batch and
shifted to maintain the effect of non linearity
"""
class CNNBatchNormLayer(object):

    def __init__(self,inputData,image_shape):
        self.input=inputData
        num_out=image_shape[-3]
        epsilon=0.01
        self.image_shape=image_shape

        gamma_values = numpy.ones((num_out,), dtype=theano.config.floatX)
        self.gamma_vals = theano.shared(value=gamma_values, borrow=True)

        beta_values = numpy.zeros((num_out,), dtype=theano.config.floatX)
        self.beta_vals = theano.shared(value=beta_values, borrow=True)

        batch_mean=T.mean(self.input,keepdims=True,axis=(0,-2,-1))
        batch_var=T.var(self.input,keepdims=True,axis=(0,-2,-1))+epsilon

        self.batch_mean=self.adjustVals(batch_mean)
        batch_var=self.adjustVals(batch_var)
        self.batch_var=T.pow(batch_var,0.5)

        batch_normalize=(inputData-self.batch_mean)/(T.pow(self.batch_var,0.5))

        if self.input.ndim==5:
            self.beta = self.beta_vals.dimshuffle('x','x', 0, 'x', 'x')
            self.gamma = self.gamma_vals.dimshuffle('x','x', 0, 'x', 'x')
        else:
            self.beta = self.beta_vals.dimshuffle('x', 0, 'x', 'x')
            self.gamma = self.gamma_vals.dimshuffle('x', 0, 'x', 'x')

        self.output=batch_normalize*self.gamma+self.beta
        #self.output=inputData-self.batch_mean

        self.params=[self.gamma_vals,self.beta_vals]


    def tileMap(self,val,prev):
        return T.tile(val,(self.image_shape[-2],self.image_shape[-1]))


    def adjustVals(self,batch_vals):
        seq=batch_vals
        #outputs_info = T.as_tensor_variable(np.asarray(0, seq.dtype))
        outputs_info=T.zeros_like(self.input[0])
        scan_result, scan_updates = theano.scan(fn=self.tileMap,
                                        outputs_info=outputs_info,
                                        sequences=seq)
        #filled_vals = theano.function(inputs=[seq], outputs=scan_result)
        #return filled_vals(seq)
        return scan_result


    def assignParams(self,gamma,beta):
        updates_gamma=(self.gamma_vals,gamma)
        updates_beta=(self.beta_vals,beta)

        assignGamma=theano.function(
            inputs=[],
            updates=[updates_gamma]
        )

        assignBeta=theano.function(
            inputs=[],
            updates=[updates_beta]
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

    def __init__(self,rng,inputData,image_shape,filter_shape,output_shape):
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

        op=T.nnet.abstract_conv.AbstractConv2d_gradInputs(output_shape,filter_shape,border_mode='half',filter_flip=False)
        self.output=op(self.W,self.input,output_shape[2:])

        self.params=[self.W,self.b]


    def assignParams(self,W,b):
        updates_W=(self.params[0],W)
        updates_b=(self.params[1],b)

        assignW=theano.function(
            inputs=[],
            updates=[updates_W]
        )

        assignb=theano.function(
            inputs=[],
            updates=[updates_b]
        )

        assignW()
        assignb()


class DeConvLayer(object):
    """
    def __init__(self,rng,inputData,image_shape,filter_shape):
        self.input=lasagne.layers.InputLayer(shape=image_shape,input_var=inputData)

        self.deConvLayer=TransposedConv2DLayer(self.input,num_filters=filter_shape[0],
        filter_size=(filter_shape[2],filter_shape[3]),nonlinearity=lasagne.nonlinearities.linear)

        self.params=self.deConvLayer.get_params()

        #self.output=self.deConvLayer.get_output_for(self.input)
        self.output=lasagne.layers.get_output(self.deConvLayer)
    """
    def __init__(self,rng,inputData,image_shape,filter_shape,output_shape):
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

        op=T.nnet.abstract_conv.AbstractConv2d_gradInputs(output_shape,filter_shape,filter_flip=False)
        self.output=op(self.W,self.input,output_shape[2:])

        self.params=[self.W,self.b]


    def assignParams(self,W,b):
        updates_W=(self.params[0],W)
        updates_b=(self.params[1],b)

        updates_W=(self.W,W)
        updates_b=(self.b,b)

        assignW=theano.function(
            inputs=[],
            updates=[updates_W]
        )

        assignb=theano.function(
            inputs=[],
            updates=[updates_b]
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
