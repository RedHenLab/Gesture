import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from theano import gof, Op, tensor, Variable, Apply
from theano.gof.null_type import NullType, null_type


def crop2d(input,offset):
    op = Crop(offset)
    output = op(input)
    return output


def fuse2d(input1,input2):
    op=FuseSum()
    output=op(input1,input2)
    return output


class Crop(Op):
    """
    Implement centre cropping
    """

    @staticmethod
    def out_shape(imgshape,offset):
        """
        computes the output shape given the image shape and the offset.
        """
        nr=imgshape[2]
        nc=imgshape[3]

        rval=list(imgshape[:-2]) + [nr-2*offset,nc-2*offset]
        return rval

    def __init__(self,offset):
        self.offset=offset

    def make_node(self,x):
        """
        This is used to make node for theano compilation.
        """
        if x.type.ndim != 4:
            raise TypeError()
        # TODO: consider restricting the dtype?
        x = tensor.as_tensor_variable(x)
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x.broadcastable[:2] + (False, False)
        out = tensor.TensorType(x.dtype, broad)
        return Apply(self, [x], [x.type()])


    def perform(self,node,inp,out):
        """
        Python code for performing the op
        """
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


    def c_headers(self):
        return ['<algorithm>']


    def c_code(self, node, name, inp, out, sub):
        """
        C code to perform the op.
        """
        x, = inp
        z, = out
        offset=self.offset
        ccode="""
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int z_r, z_c, x_rn, x_cn; // shape of the output
        int r, c; // shape of the input

        r = PyArray_DIMS(%(x)s)[2];
        c = PyArray_DIMS(%(x)s)[3];
        z_r=r-2* %(offset)s;
        z_c=c-2* %(offset)s;
        //cout<<"hello";

        // memory allocation of z if necessary
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1])
          ||(PyArray_DIMS(%(z)s)[2] != z_r)
          ||(PyArray_DIMS(%(z)s)[3] != z_c)
          )
        {
          if (%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=PyArray_DIMS(%(x)s)[0];
          dims[1]=PyArray_DIMS(%(x)s)[1];
          dims[2]=z_r;
          dims[3]=z_c;
          //TODO: zeros not necessary
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
        }
        if (z_r && z_c)
        {
            for(int b=0; b<PyArray_DIMS(%(x)s)[0]; b++){
              for(int k=0; k<PyArray_DIMS(%(x)s)[1]; k++){
                for(int i=0; i< z_r; i++){
                    for(int j=0; j<z_c; j++){
                            x_rn=i+%(offset)s;
                            x_cn=j+%(offset)s;

                            dtype_%(z)s * z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, b, k, i, j)));
                            dtype_%(x)s * x = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s, b, k, x_rn, x_cn)));
                            z[0]=x[0]  ;
                        }
                    }
                }
            }
        }
        """

        return ccode % locals()


    def grad(self,inps,out_grads):
        """
        Called by theano back prop.
        """
        x,=inps
        gz,=out_grads

        return np.array([CropGrad(self.offset)(x,gz)])



class CropGrad(Op):
    """
    Called by crop grad. Implemented in C for speed.
    """

    @staticmethod
    def out_shape(imgshape,offset):
        nr=imgshape[2]
        nc=imgshape[3]

        rval=list(imgshape[:-2]) + [nr-2*offset,nc-2*offset]
        return rval

    def __init__(self,offset):
        self.offset=offset


    def make_node(self,x,y):
        if x.type.ndim != 4:
            raise TypeError()
        if y.type.ndim != 4:
            raise TypeError()
        # TODO: consider restricting the dtype?
        x = tensor.as_tensor_variable(x)
        y = tensor.as_tensor_variable(y)
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x.broadcastable[:2] + (False, False)
        out = tensor.TensorType(x.dtype, broad)
        return Apply(self, [x,y], [out()])


    def perform(self,node,inps,out):
        x,gz=inps
        z,=out
        #z[0]=np.empty(x.shape, dtype=x.dtype)
        print "hello"
        zz=np.zeros_like(x)
        offset=self.offset
        #print x.shape

        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(offset,x.shape[2]-offset):
                    for c in xrange(offset,x.shape[3]-offset):
                        zz[n,k,r,c]=gz[n,k,r-offset,c-offset]

        z[0]=zz

        #return zz



class FuseSum(Op):
    """
    Implement addition of two frames with the same size.
    """

    @staticmethod
    def out_shape(imgshape):
        nr=imgshape[2]
        nc=imgshape[3]

        rval=list(imgshape[:-2]) + [nr,nc]
        return rval


    def __init__(self):
        pass


    def make_node(self,x,y):
        if x.type.ndim != 4:
            raise TypeError()
        if y.type.ndim != 4:
            raise TypeError()
        # TODO: consider restricting the dtype?
        x = tensor.as_tensor_variable(x)
        y = tensor.as_tensor_variable(y)
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x.broadcastable[:2] + (False, False)
        out = tensor.TensorType(x.dtype, broad)
        return gof.Apply(self, [x,y], [out()])


    def perform(self,node,inp,out):
        x = inp[0]
        y = inp[1]
        z, = out

        if len(x.shape) != 4:
            raise NotImplementedError(
                'Pool requires 4D input for now')

        if len(y.shape) != 4:
            raise NotImplementedError(
                'Pool requires 4D input for now')

        if(x.shape!=y.shape):
            raise ValueError("The shapes should match")

        z_shape=self.out_shape(x.shape)
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = np.empty(z_shape, dtype=x.dtype)
        zz = z[0]

        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(x.shape[2]):
                    for c in xrange(x.shape[3]):
                        zz[n,k,r,c]=x[n,k,r,c]+y[n,k,r,c]


    def c_headers(self):
        return ['<algorithm>']


    def c_code(self, node, name, inp, out, sub):
        x = inp[0]
        y = inp[1]

        z, = out
        ccode="""
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int z_r, z_c; // shape of the output
        int r, c; // shape of the input

        r = PyArray_DIMS(%(x)s)[2];
        c = PyArray_DIMS(%(x)s)[3];
        z_r=r;
        z_c=c;
        //cout<<"hello";

        // memory allocation of z if necessary
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1])
          ||(PyArray_DIMS(%(z)s)[2] != z_r)
          ||(PyArray_DIMS(%(z)s)[3] != z_c)
          )
        {
          if (%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=PyArray_DIMS(%(x)s)[0];
          dims[1]=PyArray_DIMS(%(x)s)[1];
          dims[2]=z_r;
          dims[3]=z_c;
          //TODO: zeros not necessary
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
        }
        if (z_r && z_c)
        {
            for(int b=0; b<PyArray_DIMS(%(x)s)[0]; b++){
              for(int k=0; k<PyArray_DIMS(%(x)s)[1]; k++){
                for(int i=0; i< z_r; i++){
                    for(int j=0; j<z_c; j++){
                            dtype_%(z)s * z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, b, k, i, j)));
                            dtype_%(x)s * x = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s, b, k, i, j)));
                            dtype_%(y)s * y = ((dtype_%(y)s*)(PyArray_GETPTR4(%(y)s, b, k, i, j)));
                            z[0]=x[0] + y[0] ;
                        }
                    }
                }
            }
        }
        """

        return ccode % locals()


    def grad(self,inps,out_grads):
        return out_grads[0],out_grads[0]
