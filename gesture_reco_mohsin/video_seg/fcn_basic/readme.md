#Fully convolution network

This code tries to simulate the FCN in Theano with the weights given in http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel

##Files

The file layers.py contains the implementation required for the full structure of the network.

The file fcn.py contains the structure of the network as described in http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel using the layers implemented in layers.py.

## Pre-requisite

To run the network, it is required to have pickle and Theano installed.

## To run

To run the code just do:
```
python fcn.py
```

##Latest updates

The code was tested on Mac OSX with 2.1 GHz processor and 8 GB RAM and average time to get output for an image as 15 seconds on CPU.
