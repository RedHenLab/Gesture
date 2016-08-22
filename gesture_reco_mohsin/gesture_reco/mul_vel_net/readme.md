#Multi velocity network

This code simulates the network given in https://arxiv.org/pdf/1603.06829v1.pdf for both training and testing using Cohn-Kanade dataset. 

##Files

The file layers.py contains the implementation required for the full structure of the network.

The file net.py contains the structure of the network using the layers implemented in layers.py.

The file video_support.py contains the functionalities required to load and save the video.

The module ck_support.py contains the code for feeding the data from CK+ dataset to the system.

The detail help for the files can be found in the docs.

## Pre-requisite

To run the network, it is required to have pickle and Theano installed. For the code to be GPU compatible, it is required that the floatX flag be set to float32.

## setting up

To set  the system for testing in the main of net.py, set the function test=True.

## To run

To run the code just do:
```
THEANO_FLAGS=floatX=float32 python fcn.py
```

To run the code for GPU do:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python fcn.py
```
