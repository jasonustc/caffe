import numpy as np
#import Image
import pdb
# Make sure that caffe is on the python path:
caffe_root = './'

import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Load the original network and extract the fully connected layers' parameters.
net = caffe.Net('./VGG_finetune.prototxt', 
        './VGG_ILSVRC_16_layers.caffemodel',
        caffe.TEST)
params = ['fc6', 'fc7', 'fc8']
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional.'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)
pdb.set_trace()

#Load the fully convolutional network to transplant the parameters.
net_full_conv = caffe.Net('./VGG_dense_test.prototxt',
        './VGG_ILSVRC_16_layers.caffemodel',
        caffe.TEST)
params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params:
    print '{} weights are {} dimensional and biases are {} \
    dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)
pdb.set_trace()

for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat #flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]
pdb.set_trace()
net_full_conv.save('./VGG_16_fc.caffemodel')

