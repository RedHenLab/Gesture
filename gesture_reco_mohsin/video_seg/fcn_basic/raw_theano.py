import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from theano import gof, Op, tensor, Variable, Apply


def crop2d(input,offset):
    op = Crop(offset)
    output = op(input)
    return output


class Crop(Op):
    """
    Implement centre cropping
    """

    @staticmethod
    def out_shape(imgshape,offset):
        nr=imgshape[2]
        nc=imgshape[3]

        rval=list(imgshape[:-2]) + [nr-2*offset,nc-2*offset]
        return rval



    def __init__(self,offset):
        self.offset=offset

    def make_node(self,x):
        if x.type.ndim != 4:
            raise TypeError()
        # TODO: consider restricting the dtype?
        x = tensor.as_tensor_variable(x)
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x.broadcastable[:2] + (False, False)
        out = tensor.TensorType(x.dtype, broad)
        return gof.Apply(self, [x], [out()])

    def perform(self,node,inp,out):
        x, = inp
        z, = out
        if len(x.shape) != 4:
            raise NotImplementedError(
                'Pool requires 4D input for now')

        z_shape=self.out_shape(x.shape,self.offset)
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = np.empty(z_shape, dtype=x.dtype)
        zz = z[0]
        offset=self.offset

        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(x.shape[2]-2*offset):
                    for c in xrange(x.shape[3]-2*offset):
                        zz[n,k,r,c]=x[n,k,r+offset,c+offset]
