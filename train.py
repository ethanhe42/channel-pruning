from __future__ import print_function
from easydict import EasyDict as edict
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

from lib.decompose import *
from lib.net import Net, load_layer, caffe_test
from lib.utils import *
from lib.worker import Worker

sys.path.insert(0, osp.dirname(__file__)+'/lib')

def step0(pt, model):
    net = Net(pt, model=model, noTF=1)
    WPQ, pt, model = net.preprocess_resnet()
    return {"WPQ": WPQ, "pt": pt, "model": model}

def step1(pt, model, WPQ, check_exist=False):
    print(pt)
    net = Net(pt, model, noTF=1)
    model = net.finalmodel(WPQ)
    if 1:
        convs = net.convs
    else:
        convs = net.convs[:-1]
        redprint("ignoring last conv!")
    if dcfgs.dic.option == 1:
        sums = net.type2names('Eltwise')[:-1]
        newsums = []
        for i in sums:
            if not i.endswith('block8_sum'):
                newsums.append(i)
        newconvs = []
        for i in convs:
            if i.endswith('_proj'):
                newconvs.insert(0,i)
            else:
                newconvs.append(i)
        convs = newsums + newconvs
    else:
        convs += net.type2names('Eltwise')[:-1]
    if dcfgs.dic.fitfc:
        convs += net.type2names('InnerProduct')
    if dcfgs.model in [cfgs.Models.xception,cfgs.Models.resnet]:
        for i in net.bns:
            if 'branch1' in i:
                convs += [i]
    net.freeze_images(check_exist=check_exist, convs=convs)
    return {"model":model}

def combine():
    net = Net(dcfgs.prototxt, dcfgs.weights)
    net.combineHP()

def c3(pt=cfgs.vgg.model,model=cfgs.vgg.weights):
    dcfgs.splitconvrelu=True
    cfgs.accname='accuracy@5'
    def solve(pt, model):
        net = Net(pt, model=model)
        net.load_frozen()
        WPQ, new_pt = net.R3()
        return {"WPQ": WPQ, "new_pt": new_pt}

    def stepend(new_pt, model, WPQ):
        net = Net(new_pt, model=model)
        net.WPQ = WPQ
        net.finalmodel(save=False)
        net.dis_memory()
        #final = net.finalmodel(WPQ, prefix='3r')
        new_pt, new_model = net.save(prefix='3c')
        print('caffe test -model',new_pt, '-weights',new_model)
        return {"final": None}
    
    worker = Worker()
    outputs = worker.do(step0, pt=pt, model=model)
    printstage("freeze")
    pt = outputs['pt']
    outputs = worker.do(step1,**outputs)
    printstage("speed", dcfgs.dic.keep)
    outputs['pt'] = mem_pt(pt)
    if 0:
        outputs = solve(**outputs)
    else:
        outputs = worker.do(solve, **outputs)
    printstage("saving")
    outputs = worker.do(stepend, model=model, **outputs)

def splitrelu():
    net = Net(dcfgs.prototxt, model=dcfgs.weights)
    print(net.seperateConvReLU())

def addbn(pt='../resnet-cifar10-caffe/resnet-56/prb_mem_bn_trainval.prototxt', model="../resnet-cifar10-caffe/resnet-56/snapshot/prb_VH_bn__iter_64000.caffemodel"):
    worker=Worker()
    def ad(pt, model):
        net = Net(pt, model=model, noTF=1)
        #net.computation()
        pt, WPQ = net.add_bn()
        return {'new_pt': pt, 'model':model, 'WPQ':WPQ}
    outs = worker.do(ad, pt=pt, model=model)
    worker.do(stepend, **outs)
    #stepend(**outs)

def compute(pt='../resnet-cifar10-caffe/resnet-56/trainval.prototxt', model="../resnet-cifar10-caffe/resnet-56/snapshot/_iter_64000.caffemodel"):
    net = Net(pt, model=model, noTF=1)
    net.computation()

def parse_args():
    parser = argparse.ArgumentParser("experiment")
    parser.add_argument('-tf', dest='tf_vis', help='tf devices', default=None, type=str)
    parser.add_argument('-caffe', dest='caffe_vis', help='caffe devices', default=None, type=str)
    parser.add_argument('-action', dest='action', help='action', default='train', type=str)
    attrs = ['dic', 'an', 'res']
    for d in attrs:
        for i in dcfgs[d]:
            parser.add_argument('-'+d+'.'+i, dest=d+'DOT'+i, help=d+'.'+i, default=None,type=str)

    for i in dcfgs:
        if i not in attrs:
            parser.add_argument('-'+i, dest=i, help=i, default=None,type=str)

    args = parser.parse_args()
    if args.tf_vis is not None: cfgs.tf_vis = args.tf_vis
    if args.caffe_vis is not None: cfgs.caffe_vis = args.caffe_vis
    for d in attrs:
        for i in dcfgs[d]:
            att = getattr(args, d+'DOT'+i)
            if att is not None:
                if 0:
                    print(d,i, att)
                dcfgs[d][i]=type(dcfgs[d][i])(att)

    for i in dcfgs:
        if i in attrs:
            continue
        att = getattr(args, i)
        if att is not None:
            dcfgs[i]=type(dcfgs[i])(att)
    
    dcfgs.Action = args.action
    if args.model is not None:
        netmodel = getattr(cfgs, args.model)
        cfgs.accname = netmodel.accname
        if args.prototxt is None:
            dcfgs.prototxt = netmodel.model
        if args.weights is None:
            dcfgs.weights = netmodel.weights
    return args

if __name__ == '__main__':

    args = parse_args()
    cfgs.set_nBatches(dcfgs.nBatches)

    dcfgs.dic.option=1

    if args.action == cfgs.Action.addbn:
        addbn(pt=dcfgs.prototxt, model=dcfgs.weights)

    elif args.action == cfgs.Action.splitrelu:
        splitrelu()

    elif args.action == cfgs.Action.c3:
        c3()
    elif args.action == cfgs.Action.combine:
        combine()
    else:
        pass
