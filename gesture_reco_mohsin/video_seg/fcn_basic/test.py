import theano
from layers import *
import numpy as np

def testBN():
    x=T.tensor4('x')
    np.random.seed(0)
    d=np.random.rand(1,2,5,5)
    img_shape=(1,2,5,5)
    bn_layer=CNNBatchNormLayer(x,img_shape)
    bn_layer.assignParams([2,1],[0,0])

    out=bn_layer.output
    print d

    testNet=theano.function(
        inputs=[x],
        outputs=[out],
        on_unused_input='warn'
    )

    output=testNet(d)
    print output

testBN()
