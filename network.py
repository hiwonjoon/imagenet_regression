BATCH_SIZE = 128
CLIP_DELTA = 10.0
LEARNING_RATE = 1
RHO = 0.90
RMS_EPSILON = 0.0001

import lasagne

from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX

import numpy as np
import theano
import theano.tensor as T 

class Net:
    def __init__(self):
        #Define network

        net = {}
        net['input'] = InputLayer((None, 3, 224, 224))
        net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2)
        net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
        net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
        net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5)
        net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
        net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1)
        net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1)
        net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1)
        net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
        net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
        net_fv = net['fc7'] 

        # Load Pretrained parameters
        import pickle

        model = pickle.load(open('../../models/vgg_cnn_s.pkl'))

        lasagne.layers.set_all_param_values(net_fv, model['values'][0:14])

        # Make a Train & a Test Function
        input1 = T.ftensor4('input1')
        input1_shared = theano.shared( np.zeros( (BATCH_SIZE, 3, 224, 224 ), dtype = np.float32 ))
        input2 = T.ftensor4('input2')
        input2_shared = theano.shared( np.zeros( (BATCH_SIZE, 3, 225, 225 ), dtype = np.float32 ))
        target_distance = T.icol('distance')
        distance_shared = theano.shared( np.zeros(( BATCH_SIZE, 1), dtype = 'int32'), broadcastable=(False,True) )
        
        # Train function
        fv1 = lasagne.layers.get_output( net_fv, input1 )
        fv2 = lasagne.layers.get_output( net_fv, input2 )

        dist = T.sum(lasagne.objectives.squared_error(fv1,fv2),axis=1) ** 0.5
        diff = target_distance - dist
        loss_before_clipped = T.mean(diff**2) ** 0.5
        diff = diff.clip(-CLIP_DELTA,CLIP_DELTA)
        loss = T.mean(diff**2) ** 0.5

        params = lasagne.layers.get_all_params(net_fv,trainable=True)
        updates = lasagne.updates.rmsprop(loss, params, LEARNING_RATE, RHO, RMS_EPSILON)

        train_func = theano.function([], [loss,loss_before_clipped], updates = updates, givens={input1:input1_shared,input2:input2_shared,target_distance:distance_shared})

        # Test function 
        fv1_determ = lasagne.layers.get_output( net_fv, input1, deterministic=True )
        fv2_determ = lasagne.layers.get_output( net_fv, input2, deterministic=True )

        dist_determ = T.sum(lasagne.objectives.squared_error(fv1_determ,fv2_determ),axis=1) ** 0.5

        test_func = theano.function([], [dist_determ,fv1_determ,fv2_determ], givens={input1:input1_shared,input2:input2_shared})

        self.net = net
        self.CLASSES = model['synset words']
        self.MEAN_IMAGE = model['mean image']
        self.input1 = input1_shared
        self.input2 = input2_shared
        self.distance = distance_shared
        self.train_func = train_func
        self.test_func = test_func

    def train(self,ims1,ims2,dist):
        self.input1.set_value(ims1)
        self.input2.set_value(ims2)
        self.distance.set_value(dist) 
        loss, loss_before_clipped = self.train_func()
        return loss, loss_before_clipped

    def test(self,ims1, ims2) :
        self.input1.set_value(ims1)
        self.input2.set_value(ims2)
        dist, fv1, fv2 = self.test_func()
        return dist, fv1, fv2
# Begin Training
#for x in xxx :
#    for k in range(BATCH_SIZE) :
#        input1_shared[k,...] = xxx
#        input2_shared[k,...] = yyy
#        distance_shared[k] = zzz
#    
#    loss, fv1, fv2 = train_func()
#    print loss
