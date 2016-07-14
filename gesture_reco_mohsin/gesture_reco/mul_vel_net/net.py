import theano
import theano.tensor as T
from layers import *

import numpy as np

# This class will be used for making three blocks in
# which videos with different velocities will be given
# as input.

# The parameters of the network are
# 1) The number of frames in the initial videos: 25 in paper
# 2) The size and stride to the four input blocks division: 9 in paper
# 3) The size and stride in the temporal domain

dtensor5=T.TensorType('float64',(False,)*5)


class DeConvBlock(object):

    def __init__(self,batch_size):
        x=dtensor5('x')
        self.x=x

        self.batch_size=batch_size
        self.params=[]

        rng = numpy.random.RandomState(23455)
        convLayer1_input_shape=[batch_size,5,3,50,50]
        convLayer1_filter_shape=[10,3,3,5,5]
        self.convlayer1=TemporalConvLayer(rng,x,convLayer1_input_shape,convLayer1_filter_shape,2)
        self.params.extend(self.convlayer1.params)

        self.deconvLayer1=TemporalDeConvLayer(rng,self.convlayer1.output,convLayer1_input_shape,convLayer1_filter_shape,2)


    def test(self,test_set_x):
        out=self.deconvLayer1.output
        batch_size=self.batch_size

        index = T.lscalar()
        testDataX=theano.shared(test_set_x)

        testDeConvNet=theano.function(
            inputs=[index],
            outputs=out,
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
            outs.append(out)

        return np.array(outs)


    def train(self,train_set_x,learning_rate,train_set_y=None):
        #lossLayer=SoftmaxWithLossLayer(self.score_Layer.output)
        #loss=T.sum(lossLayer.output)

        gparams=T.grad(T.sum(self.convlayer1.output),self.convlayer1.input)
        #updates = [
        #    (param, param - learning_rate * gparam)
        #    for param, gparam in zip(self.params, gparams)
        #]

        index = T.lscalar()
        trainDataX=theano.shared(train_set_x)

        batch_size=self.batch_size

        trainDeConvNet=theano.function(
            inputs=[index],
            outputs=[gparams],
            #updates=updates,
            on_unused_input='warn',
            givens={
                self.x :trainDataX[index * batch_size: (index + 1) * batch_size]
            },
        )

        outs=[]

        n_train_batches=int(numpy.floor(len(train_set_x)/batch_size))
        print n_train_batches
        for batch_index in range(n_train_batches):
            out=trainDeConvNet(batch_index)
            print out[0].shape



if __name__=="__main__":
    #dtensor5=T.TensorType('float64',(False,)*5)

    z=dtensor5('z')

    block=DeConvBlock(1)
    x=np.random.rand(1,5,3,50,50)
    out=block.test(x)
    #block.train(x,0.1)

    print out.shape
