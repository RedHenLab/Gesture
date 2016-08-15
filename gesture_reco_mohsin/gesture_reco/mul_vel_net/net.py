import theano
import theano.tensor as T
from layers import *
from video_support import *
from ck_support import *

import pickle
import numpy as np

# This class will be used for making three blocks in
# which videos with different velocities will be given
# as input.

# The parameters of the network are
# 1) The number of frames in the initial videos: 25 in paper
# 2) The size and stride to the four input blocks division: 9 in paper
# 3) The size and stride in the temporal domain
# 4) The factors by which to slow down the video

dtensor5=T.TensorType('float32',(False,)*5)


class DeConvBlock(object):

    def __init__(self,batch_size):

        self.x=dtensor5('x')
        # assuming that initial video is of size 25 frames
        video_frag_size=9
        x1_1_start=0
        x1_2_start=1
        x1_3_start=2
        x1_4_start=3
        self.x1_1=self.x[:,x1_1_start:x1_1_start+video_frag_size,:,:,:]
        self.x1_2=self.x[:,x1_2_start:x1_2_start+video_frag_size,:,:,:]
        self.x1_3=self.x[:,x1_3_start:x1_3_start+video_frag_size,:,:,:]
        self.x1_4=self.x[:,x1_4_start:x1_4_start+video_frag_size,:,:,:]

        self.batch_size=batch_size
        self.params=[]

        rng = numpy.random.RandomState(23455)

        convLayer1_input_shape=[batch_size,9,3,145,145]
        convLayer1_filter_shape=[96,3,3,11,11]
        conv1_temporal_stride=2
        conv1_filter_stride=3
        self.convLayer1_1=TemporalConvLayer(rng,self.x1_1,convLayer1_input_shape,
        convLayer1_filter_shape,conv1_temporal_stride,conv1_filter_stride)
        self.params.extend(self.convLayer1_1.params)

        NormLayer1_input_shape=[batch_size,4,96,45,45]
        self.NormLayer1_1=CNNBatchNormLayer(self.convLayer1_1.output,NormLayer1_input_shape)
        self.params.extend(self.NormLayer1_1.params)


        self.convLayer1_2=TemporalConvLayer(rng,self.x1_2,convLayer1_input_shape,
        convLayer1_filter_shape,conv1_temporal_stride,conv1_filter_stride)
        self.params.extend(self.convLayer1_2.params)

        self.NormLayer1_2=CNNBatchNormLayer(self.convLayer1_2.output,NormLayer1_input_shape)
        self.params.extend(self.NormLayer1_2.params)


        self.convLayer1_3=TemporalConvLayer(rng,self.x1_3,convLayer1_input_shape,
        convLayer1_filter_shape,conv1_temporal_stride,conv1_filter_stride)
        self.params.extend(self.convLayer1_3.params)

        self.NormLayer1_3=CNNBatchNormLayer(self.convLayer1_3.output,NormLayer1_input_shape)
        self.params.extend(self.NormLayer1_3.params)


        self.convLayer1_4=TemporalConvLayer(rng,self.x1_4,convLayer1_input_shape,
        convLayer1_filter_shape,conv1_temporal_stride,conv1_filter_stride)
        self.params.extend(self.convLayer1_4.params)

        self.NormLayer1_4=CNNBatchNormLayer(self.convLayer1_4.output,NormLayer1_input_shape)
        self.params.extend(self.NormLayer1_4.params)


        convLayer2_1_input=T.concatenate([self.NormLayer1_1.output,self.NormLayer1_2.output],axis=2)
        convLayer2_2_input=T.concatenate([self.NormLayer1_3.output,self.NormLayer1_4.output],axis=2)


        convLayer2_input_shape=[batch_size,4,192,45,45]
        convLayer2_filter_shape=[256,2,192,5,5]
        conv2_temporal_stride=2
        conv2_filter_stride=2
        self.convLayer2_1=TemporalConvLayer(rng,convLayer2_1_input,convLayer2_input_shape,
        convLayer2_filter_shape,conv2_temporal_stride,conv2_filter_stride)
        self.params.extend(self.convLayer2_1.params)

        NormLayer2_input_shape=[batch_size,2,256,21,21]
        self.NormLayer2_1=CNNBatchNormLayer(self.convLayer2_1.output,NormLayer2_input_shape)
        self.params.extend(self.NormLayer2_1.params)


        self.convLayer2_2=TemporalConvLayer(rng,convLayer2_2_input,convLayer2_input_shape,
        convLayer2_filter_shape,conv2_temporal_stride,conv2_filter_stride)
        self.params.extend(self.convLayer2_2.params)


        self.NormLayer2_2=CNNBatchNormLayer(self.convLayer2_2.output,NormLayer2_input_shape)
        self.params.extend(self.NormLayer2_2.params)


        convLayer3_1_input=T.concatenate([self.NormLayer2_1.output,self.NormLayer2_2.output],axis=2)


        convLayer3_input_shape=[batch_size,2,512,21,21]
        convLayer3_filter_shape=[384,2,512,3,3]
        conv3_temporal_stride=1
        conv3_filter_stride=2
        self.convLayer3_1=TemporalConvLayer(rng,convLayer3_1_input,convLayer3_input_shape,
        convLayer3_filter_shape,conv3_temporal_stride,conv3_filter_stride)
        self.params.extend(self.convLayer3_1.params)

        NormLayer3_input_shape=[batch_size,1,384,10,10]
        self.NormLayer3_1=CNNBatchNormLayer(self.convLayer3_1.output,NormLayer3_input_shape)
        self.params.extend(self.NormLayer3_1.params)


        fc1Layer_input=self.NormLayer3_1.output
        fc1Layer_input_shape=[batch_size,1,384,10,10]
        fc1Layer_filter_shape=[4096,1,384,10,10]
        fc1Layer_temporal_stride=1
        fc1Layer_filter_stride=1
        self.fc1Layer=TemporalConvLayer(rng,fc1Layer_input,fc1Layer_input_shape,
        fc1Layer_filter_shape,fc1Layer_temporal_stride,fc1Layer_filter_stride)
        self.params.extend(self.fc1Layer.params)


        fc2Layer_input=self.fc1Layer.output
        fc2Layer_input_shape=[batch_size,1,4096,1,1]
        fc2Layer_filter_shape=[4096,1,4096,1,1]
        fc2Layer_temporal_stride=1
        fc2Layer_filter_stride=1
        self.fc2Layer=TemporalConvLayer(rng,fc2Layer_input,fc2Layer_input_shape,
        fc2Layer_filter_shape,fc2Layer_temporal_stride,fc2Layer_filter_stride)
        self.params.extend(self.fc2Layer.params)


        #self.deconvLayer1=TemporalDeConvLayer(rng,self.convlayer1.output,convLayer1_input_shape,convLayer1_filter_shape,2)

        deconvLayerfc_1_temporal_stride=1
        deconvLayerfc_1_filter_stride=1
        self.deconvLayerfc_1=TemporalDeConvLayer(rng,self.fc2Layer.output,
        fc1Layer_input_shape,fc1Layer_filter_shape,deconvLayerfc_1_temporal_stride,
        deconvLayerfc_1_filter_stride)
        self.params.extend(self.deconvLayerfc_1.params)

        NormLayer_dfc_input_shape=[batch_size,1,384,10,10]
        self.NormLayer_dfc_1=CNNBatchNormLayer(self.deconvLayerfc_1.output,NormLayer_dfc_input_shape)
        self.params.extend(self.NormLayer_dfc_1.params)


        deconvLayer3_1_temporal_stride=1
        deconvLayer3_1_filter_stride=2
        self.deconvLayer3_1=TemporalDeConvLayer(rng,self.NormLayer_dfc_1.output,
        convLayer3_input_shape,convLayer3_filter_shape,deconvLayer3_1_temporal_stride,
        deconvLayer3_1_filter_stride)
        self.params.extend(self.deconvLayer3_1.params)

        NormLayer_d3_input_shape=[batch_size,2,512,21,21]
        self.NormLayer_d3_1=CNNBatchNormLayer(self.deconvLayer3_1.output,NormLayer_d3_input_shape)
        self.params.extend(self.NormLayer_d3_1.params)


        deconvLayer2_1_input=self.NormLayer_d3_1.output[:,:,0:256,:,:]
        deconvLayer2_2_input=self.NormLayer_d3_1.output[:,:,256:,:,:]

        deconvLayer2_temporal_stride=2
        deconvLayer2_filter_stride=2
        self.deconvLayer2_1=TemporalDeConvLayer(rng,deconvLayer2_1_input,
        convLayer2_input_shape,convLayer2_filter_shape,deconvLayer2_temporal_stride,
        deconvLayer2_filter_stride)
        self.params.extend(self.deconvLayer2_1.params)

        NormLayer_d2_input_shape=[batch_size,4,192,45,45]
        self.NormLayer_d2_1=CNNBatchNormLayer(self.deconvLayer2_1.output,NormLayer_d2_input_shape)
        self.params.extend(self.NormLayer_d2_1.params)


        self.deconvLayer2_2=TemporalDeConvLayer(rng,deconvLayer2_2_input,
        convLayer2_input_shape,convLayer2_filter_shape,deconvLayer2_temporal_stride,
        deconvLayer2_filter_stride)
        self.params.extend(self.deconvLayer2_2.params)

        self.NormLayer_d2_2=CNNBatchNormLayer(self.deconvLayer2_2.output,NormLayer_d2_input_shape)
        self.params.extend(self.NormLayer_d2_2.params)


        deconvLayer1_1_input=self.NormLayer_d2_1.output[:,:,0:96,:,:]
        deconvLayer1_2_input=self.NormLayer_d2_1.output[:,:,96:,:,:]
        deconvLayer1_3_input=self.NormLayer_d2_2.output[:,:,0:96,:,:]
        deconvLayer1_4_input=self.NormLayer_d2_2.output[:,:,96:,:,:]


        deconvLayer1_temporal_stride=2
        deconvLayer1_filter_stride=3
        self.deconvLayer1_1=TemporalDeConvLayer(rng,deconvLayer1_1_input,
        convLayer1_input_shape,convLayer1_filter_shape,deconvLayer1_temporal_stride,
        deconvLayer1_filter_stride)
        self.params.extend(self.deconvLayer1_1.params)


        self.deconvLayer1_2=TemporalDeConvLayer(rng,deconvLayer1_2_input,
        convLayer1_input_shape,convLayer1_filter_shape,deconvLayer1_temporal_stride,
        deconvLayer1_filter_stride)
        self.params.extend(self.deconvLayer1_2.params)


        self.deconvLayer1_3=TemporalDeConvLayer(rng,deconvLayer1_3_input,
        convLayer1_input_shape,convLayer1_filter_shape,deconvLayer1_temporal_stride,
        deconvLayer1_filter_stride)
        self.params.extend(self.deconvLayer1_3.params)


        self.deconvLayer1_4=TemporalDeConvLayer(rng,deconvLayer1_4_input,
        convLayer1_input_shape,convLayer1_filter_shape,deconvLayer1_temporal_stride,
        deconvLayer1_filter_stride)
        self.params.extend(self.deconvLayer1_4.params)



    def test(self,test_set_x):
        out=self.deconvLayer1_1.output
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

        gparams=T.grad(T.sum(self.deconvLayer1_1.output+self.deconvLayer1_2.output+
        self.deconvLayer1_3.output+self.deconvLayer1_4.output),self.params)
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        index = T.lscalar()
        trainDataX=theano.shared(train_set_x)

        batch_size=self.batch_size

        trainDeConvNet=theano.function(
            inputs=[index],
            outputs=[],
            updates=updates,
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
            #print out[0].shape



class MulVelNet(object):

    def __init__(self,batch_size):
        self.batch_size=batch_size
        self.y=dtensor5('y')


        self.block1=DeConvBlock(batch_size)
        self.block2=DeConvBlock(batch_size)
        self.block3=DeConvBlock(batch_size)

        self.params=[]
        self.params.extend(self.block1.params)
        self.params.extend(self.block2.params)
        self.params.extend(self.block3.params)

        rng = numpy.random.RandomState(23455)

        fc1_supvis_input=T.concatenate([self.block1.fc1Layer.output
        ,self.block2.fc1Layer.output,self.block3.fc1Layer.output],axis=2)
        fc1_supvis_input_shape=[batch_size,1,4096*3,1,1]
        fc1_supvis_filter_shape=[8192,1,4096*3,1,1]
        fc1_supvis_temporal_stride=1
        fc1_supvis_filter_stride=1
        self.fc1_supvis_Layer=TemporalConvLayer(rng,fc1_supvis_input,fc1_supvis_input_shape,
        fc1_supvis_filter_shape,fc1_supvis_temporal_stride,fc1_supvis_filter_stride)
        self.params.extend(self.fc1_supvis_Layer.params)


        fc2_supvis_input=self.fc1_supvis_Layer.output
        fc2_supvis_input_shape=[batch_size,1,4096*2,1,1]
        fc2_supvis_filter_shape=[4096,1,4096*2,1,1]
        fc2_supvis_temporal_stride=1
        fc2_supvis_filter_stride=1
        self.fc2_supvis_Layer=TemporalConvLayer(rng,fc2_supvis_input,fc2_supvis_input_shape,
        fc2_supvis_filter_shape,fc2_supvis_temporal_stride,fc2_supvis_filter_stride)
        self.params.extend(self.fc2_supvis_Layer.params)


        fc3_supvis_input=self.fc2_supvis_Layer.output
        fc3_supvis_input_shape=[batch_size,1,4096,1,1]
        fc3_supvis_filter_shape=[512,1,4096,1,1]
        fc3_supvis_temporal_stride=1
        fc3_supvis_filter_stride=1
        self.fc3_supvis_Layer=TemporalConvLayer(rng,fc3_supvis_input,fc3_supvis_input_shape,
        fc3_supvis_filter_shape,fc3_supvis_temporal_stride,fc3_supvis_filter_stride)
        self.params.extend(self.fc3_supvis_Layer.params)


        fc4_supvis_input=self.fc3_supvis_Layer.output
        fc4_supvis_input_shape=[batch_size,1,512,1,1]
        fc4_supvis_filter_shape=[8,1,512,1,1]
        fc4_supvis_temporal_stride=1
        fc4_supvis_filter_stride=1
        self.fc4_supvis_Layer=TemporalConvLayer(rng,fc4_supvis_input,fc4_supvis_input_shape,
        fc4_supvis_filter_shape,fc4_supvis_temporal_stride,fc4_supvis_filter_stride)
        self.params.extend(self.fc4_supvis_Layer.params)

        self.lossLayer=SoftmaxWithLossLayer(self.fc4_supvis_Layer.output,axis_select=2)
        self.fc_cost=T.sum(T.log(self.lossLayer.output)*self.y)
        #out_label=T.max(y,axis=3)



    def test(self,test_set_x):

        out=self.lossLayer.output
        #out=out_label
        batch_size=self.batch_size

        index = T.lscalar()
        testDataX=theano.shared(test_set_x)
        loader=self.getLoaderCKData()
        #print loader.getSeqs(batch_size).shape


        testDeConvNet=theano.function(
            inputs=[index],
            outputs=out,
            on_unused_input='warn',
            givens={
                #self.block_inp: theano.shared(loader.getSeqs(batch_size)),
                self.block1.x :theano.shared(loader.getSeqs(batch_size,False,True)[0]),
                self.block2.x :theano.shared(loader.getSeqs(batch_size,True,True)[0]),
                self.block3.x :theano.shared(loader.getSeqs(batch_size,True,True)[0]),
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



    def getBlockLoss(self,block1):
        block1_cost=T.sum(T.pow(block1.x1_1-block1.deconvLayer1_1.output,2))
        block1_cost+= T.sum(T.pow(block1.x1_2-block1.deconvLayer1_2.output,2))
        block1_cost+= T.sum(T.pow(block1.x1_3-block1.deconvLayer1_3.output,2))
        block1_cost+= T.sum(T.pow(block1.x1_4-block1.deconvLayer1_4.output,2))
        return block1_cost


    def processLabels(self,labels):
        prs_labels=[]
        for label in labels:
            prs_label=np.zeros((1,8,1,1))

            if not label==-1:
                prs_label[0,label-1,0,0]=1

            prs_labels.append(prs_label)

        return np.array(prs_labels ,dtype= np.float32)



    def train(self,learning_rate,training_epochs,len_train_data):
        #lossLayer=SoftmaxWithLossLayer(self.score_Layer.output)
        #loss=T.sum(lossLayer.output)
        alpha=10
        beta=1000

        block1_cost=self.getBlockLoss(self.block1)
        block2_cost=self.getBlockLoss(self.block2)
        block3_cost=self.getBlockLoss(self.block3)

        block_cost=block1_cost+block2_cost+block3_cost
        #fc_cost=T.sum(T.log(self.lossLayer.output)*train_set_y)

        loss=alpha*block_cost+alpha*self.fc_cost

        gparams=T.grad(loss,self.params)
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        index = T.lscalar()

        loader=self.getLoaderCKData()

        batch_size=self.batch_size

        trainDeConvNet=theano.function(
            inputs=[index],
            outputs=[],
            updates=updates,
            on_unused_input='warn',
            givens={
                self.block1.x :theano.shared(loader.getSeqs(batch_size,False,True)[0]),
                self.block2.x :theano.shared(loader.getSeqs(batch_size,True,True)[0]),
                self.block3.x :theano.shared(loader.getSeqs(batch_size,True,True)[0]),
                self.y: theano.shared(self.processLabels(loader.getSeqs(batch_size,True,True)[1]))
            },
        )

        outs=[]

        n_train_batches=int(numpy.floor(len_train_data/batch_size))
        print n_train_batches

        for epoch in range(training_epochs):
            for batch_index in range(n_train_batches):
                out=trainDeConvNet(batch_index)
            loader.reset()
            #print out[0].shape


    def getLoaderCKData(self):
        data_dir="/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code_Theano/gesture_data/www.consortium.ri.cmu.edu/data/ck/CK+/cohn-kanade-images"
        label_dir="/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code_Theano/gesture_data/www.consortium.ri.cmu.edu/data/ck/CK+/Emotion"
        loader=CKDataLoader(data_dir,label_dir)
        return loader


    def saveModel(self,file_name):
        pickle.dump(self.params,open(file_name,"wb"))


    def loadModel(self,file_name):
        self.params=pickle.load(open(file_name,"rb"))




if __name__=="__main__":
    #dtensor5=T.TensorType('float64',(False,)*5)

    """
    z=dtensor5('z')

    block=DeConvBlock(1)
    x=np.random.rand(1,25,3,145,145)
    #out=block.test(x)
    block.train(x,0.1)

    #print out.shape
    """

    net=MulVelNet(1)
    x=np.random.rand(1,25,3,145,145)
    y=np.random.rand(1,1,8,1,1)
    #out=net.test(x)
    net.train(0.1,1,4)
    net.saveModel("/Users/mohsinvindhani/myHome/web_stints/gsoc16/RedHen/code_Theano/mul_vel_net_weights.p")

    #print out.shape
