from __future__ import print_function
from lib.cfgs import c as dcfgs
import lib.cfgs as cfgs
import os
os.environ['JOBLIB_TEMP_FOLDER']=dcfgs.shm
import argparse
os.environ['GLOG_minloglevel'] = '3'
import os.path as osp
import pickle
import sys
from multiprocessing import Process, Queue

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

from lib.net import Net, load_layer, caffe_test
sys.path.insert(0, osp.dirname(__file__)+'/lib')

def parse_args():
    parser = argparse.ArgumentParser("experiment")
    parser.add_argument('task', choices=['flop', 'param', 'resnet'], help='task')
    parser.add_argument('-model', dest='model', help='model dir', default="", type=str)
    parser.add_argument('-weights', dest='weights', help='weights dir', default="", type=str)
    parser.add_argument('-setting', dest='setting', help='vgg xception resnet', default="vgg", type=str)
    parser.add_argument('-tf', dest='tf_vis', help='tf devices', default=None, type=str)
    parser.add_argument('-caffe', dest='caffe_vis', help='caffe devices', default=None, type=str)
    parser.add_argument('-preflop', dest='preflop', help='original flop', default=0, type=int)

    args = parser.parse_args()
    return args

def param(model, weights):
    pass

def flop(model, weights, orig=15346630656):
    setting = getattr(cfgs, args.setting)
    if model == '':
        model = setting.model
    if weights == '':
        weights = setting.weights
    orig = setting.flop
    print('orig', orig)
    net = Net(model, model=weights, noTF=1)
    after = net.computation()
    print(after * 100 / orig)

def resnet(model='/home/heyihui/ceph/resnet-imagenet-caffe/resnet_50/ResNet-50-test.prototxt', 
        weights='/home/heyihui/ceph/resnet-imagenet-caffe/resnet_50/ResNet-50-model.caffemodel'):
    net = Net(model, model=weights, noTF=1)
    after = net.rescomputation()


if __name__ == '__main__':
    args = parse_args()
    if args.tf_vis is not None: cfgs.tf_vis = args.tf_vis
    if args.caffe_vis is not None: cfgs.caffe_vis = args.caffe_vis

    kwargs = {}
    if args.preflop is not None:
        kwargs['orig'] = args.preflop

    method_name = args.task
    possibles = globals().copy()
    possibles.update(locals())
    method = possibles.get(method_name)
    if not method:
        raise NotImplementedError("Method %s not implemented" % method_name)

    method(args.model, args.weights, **kwargs)

