BATCH_SIZE = 128
#CLIP_DELTA = 10.0
LEARNING_RATE = 0.01
NUM_OF_CLASS = 1860
IM_SIZE = 224
#REGRESSION_LAMBDA = 0.0
#RHO = 0.90
#RMS_EPSILON = 0.0001

import lasagne

from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.utils import floatX

import numpy as np
import theano
import theano.tensor as T

class ClasswiseSoftmaxLayer(lasagne.layers.MergeLayer):
    def __init__(self,incomings,**kwargs) :
        super(ClasswiseSoftmaxLayer,self).__init__(incomings,**kwargs)
    def get_output_for(self, input, **kwargs) :
        eps = 1e-6
        ret = T.exp(input[0])/(T.exp(input[0])+T.exp(input[1]))
        return ret.clip(eps,1-eps)

class Net:
    def __init__(self, lr = LEARNING_RATE):
        #Define network

        net = {}
        net['input'] = InputLayer((None, 3, IM_SIZE, IM_SIZE))
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
        net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
        net['fc8_false'] = DenseLayer(net['drop7'], num_units=NUM_OF_CLASS)
        net['fc8_true'] = DenseLayer(net['drop7'], num_units=NUM_OF_CLASS)
        net['output'] = ClasswiseSoftmaxLayer((net['fc8_true'],net['fc8_false']))
        net_outputs = net['output']


        # Make a Train & a Test Function
        image = T.ftensor4('input')
        image_shared = theano.shared( np.zeros( (BATCH_SIZE, 3, IM_SIZE, IM_SIZE ), dtype = np.float32 ))
        target_prob = T.fmatrix('target_prob')
        target_prob_shared = theano.shared( np.zeros( (BATCH_SIZE, NUM_OF_CLASS), dtype = np.float32) )

        # Train function
        prob  = lasagne.layers.get_output( net_outputs, image )
        loss= lasagne.objectives.binary_crossentropy(prob, target_prob)
        loss = lasagne.objectives.aggregate(loss,target_prob+ 1/(NUM_OF_CLASS * T.ones_like(target_prob)),mode='normalized_sum')
        
        params = lasagne.layers.get_all_params(net_outputs,trainable=True)
        updates = lasagne.updates.momentum(loss, params, lr)
        #updates = lasagne.updates.rmsprop(loss, params, LEARNING_RATE, RHO, RMS_EPSILON)
        train_func = theano.function([], [loss], updates = updates, givens={image:image_shared,target_prob:target_prob_shared})

        # Test function
        prob_determ = lasagne.layers.get_output( net_outputs, image, deterministic=True )

        test_func = theano.function([], [prob_determ], givens={image:image_shared})

        self.net = net
        self.loss = loss
        self.givens = {image:image_shared,target_prob:target_prob_shared}
        self.image = image_shared
        self.target_prob = target_prob_shared
        self.train_func = train_func
        self.test_func = test_func 

    def change_lr(self,lr):
        params = lasagne.layers.get_all_params(self.net['output'],trainable=True)
        updates = lasagne.updates.momentum(self.loss, params, lr)
        #updates = lasagne.updates.rmsprop(loss, params, LEARNING_RATE, RHO, RMS_EPSILON)

        self.train_func = theano.function([], [self.loss], updates = updates, givens=self.givens)

    def train(self,ims,target_prob):
        self.image.set_value(ims)
        self.target_prob.set_value(target_prob)
        loss = self.train_func()
        return loss

    def test(self,ims) :
        self.image.set_value(ims)
        prob = self.test_func()
        return prob

    def save_model(self,filename) :
        import cPickle as pickle
        pickle.dump(lasagne.layers.get_all_param_values(self.net['output']),open(filename,'wb'))

    def load_model(self,filename) :
        import cPickle as pickle
        model = pickle.load(open(filename))
        lasagne.layers.set_all_param_values(self.net['output'], model)
