from __future__ import print_function
from datetime import datetime
import os
from lib.cfgs import c as dcfgs
import lib.cfgs as cfgs
os.environ['CUDA_VISIBLE_DEVICES'] = cfgs.caffe_vis
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
from IPython import embed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import os
from warnings import warn
import pickle


from .decompose import pca, YYT, VH_decompose, ITQ_decompose, relu, rel_error, dictionary, fc_kernel, nonlinear_fc, ax2bxc
from .builder import Net as NetBuilder
from .utils import underline, OK, FAIL, space, CHECK_EQ, shell, redprint, Timer
# if not cfgs.noTF:
from .worker import Worker
from sklearn.linear_model import Lasso,LinearRegression, MultiTaskLasso


class layertypes:
    BatchNorm="BatchNorm"
    Scale="Scale"
    ReLU = 'ReLU'
    Pooling = 'Pooling'
    Eltwise = 'Eltwise'
    innerproduct= 'InnerProduct'

class datasets:
    imagenet = 'imagenet'
    cifar10 = 'cifar10'

class kernels:
    dic='dic'
    pruning='pruning'

class Net():
    def __init__(self, pt, model=None, phase = caffe.TEST, noTF=1, accname=None, gt_pt=None, gt_model=None):
        imagenetlsit=['imagenet', 'vgg_train_val']
        for i in imagenetlsit:
            if pt.find(i) != -1 :
                cfgs.dataset='imagenet'
        if pt.find('cifar10') != -1:
            cfgs.dataset='cifar10'

        self.caffe_device()
        if cfgs.gpu:
            caffe.set_mode_gpu()
            caffe.set_device(0)
        else:
            # caffe.set_mode_cpu()
            print("using CPU caffe")
        self.net = caffe.Net(pt, phase)#, level=2) # creates net but not load weights -by Mario
        self.pt_dir = pt
        if model is not None:
            self.net.copy_from(model)
            self.caffemodel_dir = model
        else:
            self.caffemodel_dir = 'temp/model.caffemodel'
        self.net_param = NetBuilder(pt=pt) # instantiate the NetBuilder -by Mario
        self.num = None # batch size of th validation data batch size -by Mario
        self.prunedweights = 0
        self._layers = dict()
        self._bottom_names = None
        self._top_names = None
        self.data_layer = 'data'
        if len(self.type2names('MemoryData')) != 0: # what is this for? -by Mario
            self._mem = True
        else:
            self._mem = False
        self._accname = cfgs.accname
        if self._accname is None:
            self._accname = 'accuracy@5'

        if dcfgs.kernelname == kernels.dic:
            # prefix=str(int(100*dcfgs.dic.keep))
            if dcfgs.dic.afterconv:
                self.kernel = self.grplasso_kernel
            else:
                self.kernel = self.dictionary_kernel
        else:
            self.kernel = self.pruning_kernel

        self.acc=[]
        self._protocol = 4 # None
        if gt_pt is not None:
            print("using gt model")
            self.gt_net = caffe.Net(gt_pt, phase)
            self.gt_net.copy_from(gt_model)
        self._points_dict_name = cfgs._points_dict_name
        if 0: self.show_acc('init')

        self.WPQ={} # stores pruned values, which will be saved to caffemodel later (since Net couldn't be dynamically changed) -by Mario
        self.nonWPQ = {}
        self.bottoms2ch = []
        self.bnidx = []

        self.convs= self.type2names()  # convs contains a list of strings -by Mario
        self.spation_convs = []
        self.nonsconvs = []
        for c in self.convs:
            if self.conv_param(c).group != 1:
                self.spation_convs.append(c)
            else:
                self.nonsconvs.append(c)
        self.relus = self.type2names(layer_type='ReLU')
        self.bns = self.type2names(layer_type='BatchNorm')
        self.affines = self.type2names(layer_type='Scale')
        self.pools = self.type2names(layer_type='Pooling')
        self.sums = self.type2names('Eltwise')
        self.innerproduct = self.type2names('InnerProduct')

    def tf_device(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = cfgs.tf_vis

    def caffe_device(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = cfgs.caffe_vis

    @property
    def _frozen(self):
        if cfgs.frozenname is None:
            frozenname = 'frozen' + str(dcfgs.nBatches)
        else:
            frozenname = cfgs.frozenname
        return osp.join(osp.split(self.pt_dir)[0], frozenname+".pickle")

    @property
    def top_names(self):
        if self._top_names is None:
            self._top_names = self.net.top_names
        return self._top_names

    @property
    def bottom_names(self):
        if self._bottom_names is None:
            self._bottom_names = self.net.bottom_names
        return self._bottom_names

    def layer_bottom(self, name):
        return self.net_param.layer_bottom(name)

    def _save(self, new_name, orig_name, prefix='acc'):
        if new_name is None:
            # avoid overwrite
            path, name = osp.split(orig_name)
            new_name = osp.join(path, underline(prefix, name))
        else:
            print("overwriting", new_name)
        return new_name


    def save_pt(self, new_name=None, **kwargs):
        new_name = self._save(new_name, self.pt_dir, **kwargs)
        self.net_param.write(new_name)
        return new_name

    def save_caffemodel(self, new_name=None, **kwargs):
        new_name = self._save(new_name, self.caffemodel_dir, **kwargs)
        self.net.save(new_name)
        return new_name

    def save(self, new_pt=None, new_caffemodel=None, **kwargs):
        return self.save_pt(new_pt, **kwargs), self.save_caffemodel(new_caffemodel, **kwargs)

    def param(self, name):
        if name in self.net.params:
            return self.net.params[name]
        else:
            raise Exception("no this layer")

    def blobs(self, name, gt=False):
        if gt:
            return self.gt_net.blobs[name]
        return self.net.blobs[name]


    def forward(self, gt=False):
        if gt:
            return self.gt_net.forward()
        if dcfgs.data == cfgs.Data.pro: # The forward pass can use another data format, not lmdb. For what? -by Mario
            # print("use datapro")
            self.dp.forward(True)
            imgs = self.dp["image"][0][0].clone_data("nchw")
            n_imgs = imgs.shape[0]
            labels = self.dp["label"][0][0].clone_data().squeeze().reshape((-1,1,1,1))
            self.net.set_input_arrays(imgs, labels)

        ret = self.net.forward()
        self.cum_acc(ret)
        return ret

    def param_w(self, name):
        return self.param(name)[0]

    def param_b(self, name):
        return self.param(name)[1]

    def param_data(self, name):
        return self.param_w(name).data

    def param_b_data(self, name):
        return self.param_b(name).data

    def set_param_data(self, name, data):
        if isinstance(name, tuple):
            self.param(name[0])[name[1]].data[...] = data.copy()
        else:
            self.param_w(name).data[...] = data.copy()

    def set_param_b(self, name, data):
        self.param_b(name).data[...] = data.copy()

    def ch_param_data(self, name, data):
        if isinstance(name, tuple):
            if name[1] == 0:
                self.ch_param_data(name[0], data)
            elif name[1] == 1:
                self.ch_param_b(name[0], data)
            else:
                NotImplementedError
        else:
            self.param_reshape(name, data.shape)
            self.param_w(name).data[...] = data.copy()

    def ch_param_b(self, name, data):
        self.param_b_reshape(name, data.shape)
        self.param_b(name).data[...] = data.copy()

    def param_shape(self, name):
        return self.param_data(name).shape

    def param_b_shape(self, name):
        return self.param_b_data(name).shape

    def param_reshape(self, name, shape):
        self.param_w(name).reshape(*shape)

    def param_b_reshape(self, name, shape):
        self.param_b(name).reshape(*shape)

    def data(self, name='data', **kwargs):
        return self.blobs_data(name, **kwargs)

    def label(self, name='label', **kwargs):
        return self.blobs_data(name, **kwargs)


    def blobs_data(self, name, **kwargs):
        return self.blobs(name, **kwargs).data

    def blobs_type(self, name):
        return self.blobs_data(name).dtype

    def blobs_shape(self, name):
        return self.blobs_data(name).shape

    def blobs_reshape(self, name, shape):
        return self.blobs(name).reshape(*shape)

    def blobs_num(self, name):
        if self.num is None:
            self.num = self.blobs(name).num
        return self.num

    def blobs_count(self, name):
        return self.blobs(name).count

    def blobs_height(self, name):
        return self.blobs(name).height
    def blobs_channels(self, name):
        return self.blobs(name).channels

    def blobs_width(self, name):
        return self.blobs(name).width

    def blobs_CHW(self, name):
        return self.blobs_count(name) / self.blobs_num(name)

    # =============== protobuf ===============
    def get_layer(self, conv):
        """return self.net_param.layer[conv][0]"""
        return self.net_param.layer[conv][0]

    def conv_param_stride(self, conv):
        stride = self.conv_param(conv).stride
        if len(stride) == 0:
            return 1
        else:
            assert len(stride) == 1
            return stride[0]

    def conv_param_pad(self, conv):
        pad = self.conv_param(conv).pad
        assert len(pad) == 1
        return pad[0]

    def conv_param_kernel_size(self, conv):
        kernel_size = self.conv_param(conv).kernel_size
        assert len(kernel_size) == 1
        return kernel_size[0]

    def conv_param_num_output(self, conv):
        return self.conv_param(conv).num_output

    def net_param_layer(self, conv):
        """return self.net_param.layer[conv]"""
        return self.net_param.layer[conv]

    def conv_param(self, conv):
        return self.get_layer(conv).convolution_param

    def set_conv(self, conv, num_output=0, new_name=None, pad_h=None, pad_w=None, kernel_h=None, kernel_w=None, stride=None, bias=None,group=None):
        conv_param = self.conv_param(conv)
        if num_output != 0:
            conv_param.num_output = type(conv_param.num_output)(num_output)
        if pad_h is not None:
            while len(conv_param.pad):
                conv_param.pad.remove(conv_param.pad[0])
            conv_param.pad.append(pad_h)
            conv_param.pad.append(pad_w)
        if kernel_h is not None:
            while len(conv_param.kernel_size):
                conv_param.kernel_size.remove(conv_param.kernel_size[0])
            conv_param.kernel_size.append(kernel_h)
            conv_param.kernel_size.append(kernel_w)

        if stride is not None:
            while len(conv_param.stride):
                conv_param.stride.remove(conv_param.stride[0])
            for i in stride:
                conv_param.stride.append(i)

        if bias is not None:
            conv_param.bias_term = bias

        if group is not None:
            conv_param.group = group

        if new_name is not None:
            self.net_param.ch_name(conv, new_name)

    # =============== data ===============
    def memory_preload(self, i=None):
        """randomly load data into memory"""
        if i is None:
            i = np.random.randint(dcfgs.nBatches_fc if dcfgs.dic.fitfc else dcfgs.nBatches)
        self.net.set_input_arrays(self._points_dict[(i, 0)], self._points_dict[(i, 1)])

    def usexyz(self, train=True):
        dcfgs.data = cfgs.Data.pro
        import datapro
        if train:
            provider = "provider_config_train.txt"
        else:
            provider = "provider_config_val.txt"
        self.dp = datapro.DataProvider.create_from_file(provider, "provider_cfg")
        self.dp.init()

    def extract_features(self, names=[], nBatches=None, points_dict=None, save=False):
        assert nBatches is None, "deprecate"
        nBatches = dcfgs.nBatches
        nPointsPerLayer=dcfgs.nPointsPerLayer
        if not isinstance(names, list):
            names = [names]
        inner = False
        if len(names)==1: # if we pass only 1 name, then we are operating on FC layers? -by Mario
            for top in self.innerproduct:
                if names[0] in self.bottom_names[top]:
                    inner = True
                    nBatches = dcfgs.nBatches_fc
                    break

        DEBUG = False

        pads = dict()
        shapes = dict()
        feats_dict = dict()
        def set_points_dict(name, data):
            assert name not in points_dict
            points_dict[name] = data
        dcfgs.data = cfgs.Data.lmdb  # I think this disables the use of Data.pro type of data -by Mario
        if save:
            if points_dict is None:
                frozen_points = False
                points_dict = dict()
                if 0 and self._mem: self.usexyz()

                set_points_dict("nPointsPerLayer", nPointsPerLayer)
                set_points_dict("nBatches", nBatches)
            else:
                frozen_points = True
                if nPointsPerLayer != points_dict["nPointsPerLayer"] or nBatches != points_dict["nBatches"]:
                    print("overwriting nPointsPerLayer, nBatches with frozen_points")

                nPointsPerLayer = points_dict["nPointsPerLayer"]
                nBatches = points_dict["nBatches"]

        assert len(names) > 0

        nPicsPerBatch = self.blobs_num(names[0])
        nFeatsPerBatch = nPointsPerLayer  * nPicsPerBatch
        print("run for", dcfgs.nBatches, "batches", "nFeatsPerBatch", nFeatsPerBatch)
        nFeats = nFeatsPerBatch * nBatches

        for name in names:

            """avoiding X out of bound"""
            shapes[name] = (self.blobs_height(name), self.blobs_width(name))

            if inner or len(self.blobs_shape(name))==2 or ( shapes[name][0] == 1 and shapes[name][1] == 1):
                if 0: print(name)
                chs = self.blobs_channels(name)
                if len(self.blobs_shape(name)) == 4:
                    chs*=shapes[name][0]*shapes[name][1]
                feats_dict[name] = np.ndarray(shape=(nPicsPerBatch * dcfgs.nBatches_fc,chs )) # This dict holds an entry for each conv layer Each dictionary entry will have 5000 rows,
            else:                                                                             #  each holding 1 point per layers channel (e.g. conv1_1 has 64 channels, then the shape of
                feats_dict[name] = np.ndarray(shape=(nFeats, self.blobs_channels(name)))      # feat_dict is (5000,64)
            print("Extracting", name, feats_dict[name].shape) # for a standard run,  names is a list with the conv layers: name = convs -by Mario
        idx = 0
        fc_idx = 0
        if save:
            if not frozen_points:
                set_points_dict("data", self.data().shape)
                set_points_dict("label", self.label().shape)


        runforn = dcfgs.nBatches_fc if dcfgs.dic.fitfc else dcfgs.nBatches
        for batch in range(runforn):
            if save:
                if not frozen_points:
                    self.forward()
                    set_points_dict((batch, 0), self.data().copy())
                    set_points_dict((batch, 1), self.label().copy())

                else:
                    self.net.set_input_arrays(points_dict[(batch, 0)], points_dict[(batch, 1)])
                    self.forward()
            else:
                self.forward()

            for name in names:
                # pad = pads[name]
                shape = shapes[name]
                feat = self.blobs_data(name)
                if 0: print(name, self.blobs_shape(name))
                if inner or len(self.blobs_shape(name))==2 or (shape[0] == 1 and shape[1] == 1):
                    feats_dict[name][fc_idx:(fc_idx + nPicsPerBatch)] = feat.reshape((self.num, -1))
                    continue
                if batch >= dcfgs.nBatches and name in self.convs:
                    continue
                # TODO!!! different patch for different image per batch
                if save:
                    if not frozen_points or (batch, name, "randx") not in points_dict:
                        #embed()
                        randx = np.random.randint(0, shape[0]-0, nPointsPerLayer)
                        randy = np.random.randint(0, shape[1]-0, nPointsPerLayer)
                        if dcfgs.dic.option == cfgs.pruning_options.resnet:
                            branchrandxy = None
                            branch1name = '_branch1'
                            branch2cname = '_branch2c'
                            if name in self.sums:
                                #embed()
                                nextblock = self.sums[self.sums.index(name)+1]
                                nextb1 = nextblock + branch1name
                                if not nextb1 in names:
                                    # the previous sum and branch2c will be identical
                                    branchrandxy = nextblock + branch2cname
                            elif name in self.bns:
                                if dcfgs.model == cfgs.Models.xception:
                                    branchrandxy = 'interstellar' + name.split('bn')[1].split('_')[0] + branch2cname
                                elif dcfgs.model == cfgs.Models.resnet:
                                    branchrandxy = 'res' + name.split('bn')[1].split('_')[0] + branch2cname
                                    #print("correpondance", branchrandxy)
                            if branchrandxy is not None:
                                if 0: print('pointsdict of', branchrandxy, 'identical with', name)
                                randx = points_dict[(batch, branchrandxy , "randx")]
                                randy = points_dict[(batch, branchrandxy , "randy")]

                        if name.endswith('_conv1') and dcfgs.dic.option == 1:
                            if DEBUG: redprint("this line executed becase dcfgs.dic.option is 1 [net.extract_features()]")
                            fsums = ['first_conv'] + self.sums
                            blockname = name.partition('_conv1')[0]
                            nextb1  = fsums[fsums.index(blockname+'_sum')-1]
                            branch1name = blockname + '_proj'
                            if branch1name in self.convs:
                                nextb1 = branch1name
                            randx = points_dict[(batch, nextb1 , "randx")]
                            randy = points_dict[(batch, nextb1 , "randy")]

                        set_points_dict((batch, name, "randx"), randx.copy())
                        set_points_dict((batch, name, "randy"), randy.copy())

                    else:
                        randx = points_dict[(batch, name, "randx")]
                        randy = points_dict[(batch, name, "randy")]
                else:
                    randx = np.random.randint(0, shape[0]-0, nPointsPerLayer)
                    randy = np.random.randint(0, shape[1]-0, nPointsPerLayer)

                for point, x, y in zip(range(nPointsPerLayer), randx, randy):

                    i_from = idx+point*nPicsPerBatch
                    try:
                        feats_dict[name][i_from:(i_from + nPicsPerBatch)] = feat[:,:,x, y].reshape((self.num, -1))
                    except:
                         print('total', runforn, 'batch', batch, 'from', i_from, 'to', i_from + nPicsPerBatch)
                         raise Exception("out of bound")
                if DEBUG:
                    embed()
            idx += nFeatsPerBatch
            fc_idx += nPicsPerBatch

        dcfgs.data = cfgs.Data.lmdb
        self.clr_acc()
        if save:
            if frozen_points:
                if points_dict is not None:
                    return feats_dict, points_dict
                return feats_dict
            else:
                return feats_dict, points_dict
        else:
            return feats_dict

    def extract_XY(self, X, Y, DEBUG = False, w1=None):
        """
        given two conv layers, extract X (n, c0, ks, ks), given extracted Y(n, c1, 1, 1)
        NOTE only support: conv(X) relu conv(Y)

        Return:
            X feats of size: N C h w
        """
        pad = self.conv_param_pad(Y)
        kernel_size = self.conv_param_kernel_size(Y)
        half_kernel_size = int(kernel_size/2)
        if w1 is not None:
            gw1=True
            x_pad = self.conv_param_pad(w1)
            x_ks = self.conv_param_kernel_size(w1)
            x_hks = int(x_ks/2)
        else:
            # assert x_hks == 0
            gw1=False
        stride = self.conv_param_stride(Y)

        print("Extracting X", X, "From Y", Y, 'stride', stride)



        X_h = self.blobs_height(X)
        X_w = self.blobs_width(X)
        Y_h = self.blobs_height(Y)
        Y_w = self.blobs_width(Y)

        def top2bottom(x, padded=1):
            """
            top x to bottom x0
            NOTE assume feature map padded
            """
            if padded:
                if gw1:
                    return half_kernel_size + x_hks + stride * x
                return half_kernel_size + stride * x
            return half_kernel_size - pad + stride * x

        """avoiding X out of bound"""
        # Y_pad = 0
        # while top2bottom(Y_pad) - half_kernel_size < 0:
        #     Y_pad += 1

        def y2x(y, **kwargs):
            """
            top y to bottom x patch
            """
            x0 = top2bottom(y, **kwargs)

            # return range(x0 - half_kernel_size, x0 + half_kernel_size + 1)
            if gw1:
                return x0 - half_kernel_size - x_hks, x0 + half_kernel_size + x_hks + 1
            return x0 - half_kernel_size, x0 + half_kernel_size + 1

        def bottom2top(x0):
            NotImplementedError
            return (x0 - half_kernel_size + pad) / stride


        shape = (Y_h, Y_w)

        idx = 0

        nPointsPerLayer = self._points_dict["nPointsPerLayer"]
        nBatches = self._points_dict["nBatches"]
        if gw1:
            nPicsPerBatch = self.blobs_num(X)
            nFeatsPerBatch = nPointsPerLayer  * nPicsPerBatch
            nFeats = nFeatsPerBatch * nBatches
            feat_h = (kernel_size+x_ks) - 1
            assert kernel_size % 2 == 1
            assert x_ks % 2 == 1
            feats_dict = np.ndarray(shape=(nFeats, self.blobs_channels(X), feat_h ,feat_h ))
            pass
        else:
            nPicsPerBatch = self.blobs_num(X) * kernel_size * kernel_size
            nFeatsPerBatch = nPointsPerLayer  * nPicsPerBatch
            nFeats = nFeatsPerBatch * nBatches

            feats_dict = np.ndarray(shape=(nFeats, self.blobs_channels(X)))
        X_shape = self.blobs_shape(X)

        feat_pad = pad if not gw1 else pad + x_pad
        if not self._mem:
            redprint("XY not corresponded")
        for batch in range(nBatches):
            if 0: print("done", batch, '/', nBatches)
            if self._mem:
                self.net.set_input_arrays(self._points_dict[(batch, 0)], self._points_dict[(batch, 1)])

            self.forward()

            # padding

            feat = np.zeros((X_shape[0], X_shape[1], X_shape[2] + 2 * feat_pad, X_shape[3] + 2 * feat_pad), dtype=self.blobs_type(X))
            feat[:, :, feat_pad:X_shape[2] + feat_pad, feat_pad:X_shape[3] + feat_pad] = self.blobs_data(X).copy()

            randx = self._points_dict[(batch, Y, "randx")]
            randy = self._points_dict[(batch, Y, "randy")]

            for point, x, y in zip(range(nPointsPerLayer), randx, randy):

                i_from = idx+point*nPicsPerBatch
                """n hwc"""

                x_start, x_end = y2x(x)
                y_start, y_end = y2x(y)
                if gw1:
                    if 0:
                        try:
                            feats_dict[i_from:(i_from + nPicsPerBatch)] = \
                            feat[:, :, x_start:x_end, y_start:y_end].copy()
                        except:
                            embed()
                    else:
                        feats_dict[i_from:(i_from + nPicsPerBatch)] = \
                        feat[:, :, x_start:x_end, y_start:y_end].copy()

                else:
                    feats_dict[i_from:(i_from + nPicsPerBatch)] = \
                    np.moveaxis(feat[:, :, x_start:x_end, y_start:y_end], 1, -1).reshape((nPicsPerBatch, -1))

            if DEBUG:
                # sanity check using relu(WX + B) = Y
                # embed()
                # # W: chw n
                # W = self.param_data(X).reshape(self.param_shape(X)[0], -1).T
                # # b
                # b = self.param_b_data(X)
                # n hwc
                bottom_X = feat[:, :, x_start:x_end, y_start:y_end].reshape((self.num, -1))

                # W2: chw n
                W2 = self.param_data(Y).reshape(self.param_shape(Y)[0], -1).T
                # b2
                b2 = self.param_b_data(Y)

                fake = relu(bottom_X).dot(W2) + b2

                # n c
                real = self.blobs_data(Y)[:,:,x, y].reshape((self.num, -1))

                CHECK_EQ(fake, real)

            idx += nFeatsPerBatch

        self.clr_acc()
        return feats_dict

    def extract_layers(self, names=[], nBatches=30, points_dict=None, gt=False):

        if not isinstance(names, list):
            names = [names]

        DEBUG = False
        feats_dict = dict()

        def set_points_dict(name, data):
            assert name not in points_dict
            points_dict[name] = data

        # extract_layers saves by default -by Mario
        if points_dict is None:
            frozen_points = False
            points_dict = dict()
            set_points_dict("nBatches", nBatches)
        else:
            frozen_points = True
            if nBatches != points_dict["nBatches"]:
                warn("overwriting nBatches with frozen_points")

            nBatches = points_dict["nBatches"]

        assert len(names) > 0

        nPicsPerBatch = self.blobs_num(names[0])
        nFeats = nPicsPerBatch * nBatches

        for name in names:
            feats_dict[name] = np.ndarray(shape=[nFeats] + list(self.blobs_shape(name)[1:]))
            print("Extracting", name, feats_dict[name].shape)

        idx = 0
        if not frozen_points:
            set_points_dict("data", self.data().shape)
            set_points_dict("label", self.label().shape)

        for batch in range(nBatches):
            if not frozen_points:
                print("done", batch, '/', nBatches)
                self.forward(gt=gt) # gt seems to be related to the execution of single-layer pruning -by Mario
                set_points_dict((batch, 0), self.data(gt=gt).copy())
                set_points_dict((batch, 1), self.label(gt=gt).copy())

            else:
                print("Done", batch, '/', nBatches)
                self.net.set_input_arrays(points_dict[(batch, 0)], points_dict[(batch, 1)])
                self.forward()

            i_from = batch*nPicsPerBatch
            for name in names:
                feat = self.blobs_data(name, gt=gt)
                feats_dict[name][i_from:(i_from + nPicsPerBatch)] = feat.copy()

                if DEBUG:
                    embed()

        if frozen_points:
            return feats_dict
        return feats_dict, points_dict


    def freeze_images(self, check_exist=False, convs=None, **kwargs):
        if cfgs.layer: # flag for pruning single layer -by Mario
            frozen = self._frozen_layer # code for def _frozen_layer() -using @property- is not difined -by Mario
            if check_exist: # THIS CODE WAS NOT IMPLEMENTED YET - by Mario
                pass
            else:
                pass
        else:
            frozen = self._frozen
            if check_exist:
                if osp.exists(frozen):
                    print("Exists", frozen)
                    return frozen

        if convs is None:
            convs = self.type2names()
        if cfgs.layer:
            feats_dict, points_dict = self.extract_layers(names=convs, **kwargs)
        else:
            feats_dict, points_dict = self.extract_features(names=convs, save=1, **kwargs)

        data_layer = self.data_layer
        if len(self.net_param_layer(data_layer)) == 2:
            self.net_param.net.layer.remove(self.get_layer(data_layer))

        # we will prepare the data layer to run with MemoryData type
        i = self.get_layer(data_layer)
        i.type = "MemoryData"
        i.memory_data_param.batch_size = points_dict['data'][0]
        i.memory_data_param.channels = points_dict['data'][1]
        i.memory_data_param.height = points_dict['data'][2]
        i.memory_data_param.width = points_dict['data'][3]
        i.ClearField("transform_param")
        i.ClearField("data_param")
        i.ClearField("include")

        print("wrote memory data layer to", self.save_pt(prefix="mem"))
        print("freezing imgs to", frozen)
        if cfgs.layer:
            def subfile(filename):
                return osp.join(frozen, filename)
            with open(subfile(self._points_dict_name), 'wb') as f:
                print("dumping points_dict")
                pickle.dump(points_dict, f, protocol=self._protocol)
            for conv in convs:
                with open(subfile(conv), 'wb') as f:
                    print("dumping "+conv)
                    pickle.dump(feats_dict[conv], f, protocol=self._protocol)

        else:
            with open(frozen, 'wb') as f:
                pickle.dump([feats_dict, points_dict], f, protocol=self._protocol)

        return frozen

    def dis_memory(self):
        data_layer = self.data_layer
        i = self.get_layer(data_layer)
        i.ClearField("memory_data_param")
        i.type = "Data"
        if cfgs.dataset=='imagenet':
            i.transform_param.crop_size = 224
            i.transform_param.mirror = False
            i.transform_param.mean_value.extend([104.0,117.0,123.0])
            i.data_param.source = cfgs.imagenet_val
            i.data_param.batch_size = 32
            i.data_param.backend = i.data_param.LMDB
        elif cfgs.dataset=='cifar10':
            i.transform_param.scale = .0078125
            i.transform_param.mean_value.extend([128])
            i.transform_param.mirror = False
            i.data_param.source = cfgs.cifar10_val
            i.data_param.batch_size = 128
            i.data_param.backend = i.data_param.LMDB

        else:
            assert False

        ReLUs = self.type2names("ReLU")
        Convs = self.type2names()

        for r in ReLUs:
            if self.top_names[r][0] == r:
                assert len(self.bottom_names[r]) == 1
                conv = self.bottom_names[r][0]
                self.net_param.ch_top(r, conv, r)
                for i in self.net_param.layer:
                    if i != r:
                        self.net_param.ch_bottom(i, conv, r)

    def load_frozen(self, DEBUG=False, feats_dict=None, points_dict=None):
        if feats_dict is not None:
            print("loading imgs from memory")
            self._feats_dict = feats_dict
            self._points_dict = points_dict
            return

        if cfgs.layer:
            def subfile(filename):
                return osp.join(self._frozen_layer, filename)

            with open(subfile(self._points_dict_name), 'rb') as f:
                self._points_dict = pickle.load(f)

            convs = self.type2names()
            self._feats_dict = dict()
            for conv in convs:
                filename = subfile(conv)
                if osp.exists(filename):
                    with open(filename, 'rb') as f:
                        self._feats_dict[conv] = pickle.load(f)
        else:
            frozen = self._frozen
            print("loading imgs from", frozen)
            with open(frozen, 'rb') as f:
                self._feats_dict, self._points_dict = pickle.load(f)

            if DEBUG:
                convs = self.type2names()
                feats_dict = self.extract_features(convs, points_dict=self._points_dict, save=1)
                print("feats_dict", feats_dict)
                print("self._feats_dict", self._feats_dict)
                embed()
                for i in feats_dict:
                    for x, y in zip(np.nditer(self._feats_dict[i]), np.nditer(feats_dict[i])):
                        assert  x == y
                OK("frozen         ")
        print("loaded")

    def type2names(self, layer_type='Convolution'):
        if layer_type not in self._layers:
            self._layers[layer_type] = self.net_param.type2names(layer_type)
        return self._layers[layer_type]


    def insert(self, bottom, name=None, layer_type="Convolution", bringforward=True, update_nodes=None, bringto=None, **kwargs):
        self.net_param.set_cur(bottom)
        if layer_type=="Convolution":
            # insert
            self.net_param.Convolution(name, bottom=[bottom], **kwargs)
            # clone previous layer
            if "stride" not in kwargs:
                new_conv_param = self.conv_param(name)
                while len(new_conv_param.stride):
                    new_conv_param.stride.remove(new_conv_param.stride[0])
                for i in self.conv_param(bottom).stride:
                    new_conv_param.stride.append(i)

            # update input nodes for others
            if update_nodes is None:
                update_nodes = self.net_param.layer
            for i in update_nodes:
                if i == name:
                    continue
                self.net_param.ch_bottom(i, name, bottom)
            # for i, bot in self.bottom_names.items():
            #     if bottom in bot:
            #         assert len(bot) == 1, "only support single pass"
            #         self.net_param.ch_bottom(i, name, bottom)
            if bringforward:
                if bringto is not None:
                    bottom = bringto
                self.net_param.bringforward(bottom)

        elif layer_type=="BatchNorm":
            assert name is None
            assert len(kwargs)==0
            sname = self.net_param.Scale(None, **kwargs)
            self.net_param.bringforward(bottom)
            bnname = self.net_param.BatchNorm(None, **kwargs)
            self.net_param.bringforward(bottom)
            #self.net_param.bringforward(bottom)
            return bnname, sname




    def remove(self, name, inplace=False):
        self.net_param.rm_layer(name, inplace)

    def accuracy(self, times=30):
        if times != 30:
            raise Exception("no use")
        if dcfgs.ntest:
            times = dcfgs.ntest
        else:
            times = dcfgs.nBatches
        acc = []
        for i in range(times):
            if self._mem:
                if times == dcfgs.nBatches:
                    self.memory_preload(i)
                else:
                    self.memory_preload()
            res = self.forward()
            acc.append(float(res[self._accname]))

        return np.mean(acc) #, 'std', np.std(acc)

    def cum_acc(self, res):
        self.acc.append(float(res[self._accname]))

    def clr_acc(self, show=True):
        self.currentacc = np.mean(self.acc)
        if show:
            print('Acc {:7.3f}'.format(self.currentacc*100))
        self.acc = []

    def show_acc(self, name, **kwargs):
        print("show_acc depracate")
        # print('Solved {:40s} Acc {:7.3f}'.format(name, self.accuracy(**kwargs)))

    def finalmodel(self, WPQ=None, **kwargs): # the prefix for the name of the saved model is added by self.linear() -by Mario
        """ load weights into caffemodel"""
        if WPQ is None:
            WPQ = self.WPQ
        return self.linear(WPQ, **kwargs)

    def infer_pad_kernel(self, W, origin_name):
        num_output, _, kernel_h, kernel_w = W.shape
        assert kernel_h in [3,1]
        assert kernel_w in [3,1]
        pad_h = 1 if kernel_h == 3 else 0
        pad_w = 1 if kernel_w == 3 else 0
        stride = self.conv_param(origin_name).stride
        if len(stride) == 1:
            pass
        elif len(stride) == 0:
            stride = [1]
        else:
            NotImplementedError
        if stride[0] == 1:
            pass
        elif stride[0] == 2:
            stride = [stride[0] if pad_h else 1, stride[0] if pad_w else 1]
            # stride = [1 if pad_h else stride[0], 1 if pad_w else stride[0]]
            warn("stride 2 decompose dangerous")
        else:
            NotImplementedError
        return {"pad_h":pad_h, "pad_w":pad_w, "kernel_h":kernel_h, "kernel_w":kernel_w, "num_output":num_output, "stride":stride}
    # =========algorithms=========

    def linear(self, WPQ, prefix='VH', save=True,DEBUG=0):
        for i, j in WPQ.items():
            if save:
                self.set_param_data(i, j)
            else:
                self.ch_param_data(i, j)
        if save:
            return self.save_caffemodel(prefix=prefix)


    def add_bn(self, ids=True):
        forbid = ['_id', '_proj']
        # loop over all samples
        bs = self.blobs_shape('data')[0]
        # self.usexyz(False)
        # self.dp.dataset_size
        iters = 50000 // bs
        means ={}
        variances = {}
        for i in range(iters):
            print(i,iters)
            self.forward()
            for conv in self.nonsconvs:
                if conv not in means:
                    means[conv]=[]
                    variances[conv] = []
                else:
                    means[conv].append(self.blobs_data(conv).mean((0,2,3)))
                    variances[conv].append(self.blobs_data(conv).var((0,2,3)))
        for r in self.relus:
            if self.top_names[r][0] == r:
                conv = self.bottom_names[r][0]
                if conv not in self.convs:
                    continue
                self.net_param.ch_top(r, conv, r)
                for i in self.net_param.layer:
                    if i != r:
                        self.net_param.ch_bottom(i, conv, r)

        for conv in self.nonsconvs:
            skip=False
            if forbid[0] in conv:
                # prune ids
                self.remove(conv)
                print("remove", conv)
                skip = True
            if forbid[1] in conv:
                skip = True
            if skip:
                continue
            bn, scal = self.insert(bottom=conv, layer_type=layertypes.BatchNorm)
            self.WPQ[(scal,0)] = np.array(variances[conv]).mean(0)**.5
            self.WPQ[(scal,1)] = np.array(means[conv]).mean(0)

        pt = self.save_pt(prefix = 's')
        print("ready to train", pt)
        return pt, self.WPQ

    def layercomputation(self, conv, channels=1., outputs=1.):
        bottom = self.bottom_names[conv]
        assert len(bottom) == 1
        bottom = bottom[0]
        s = self.blobs_shape(bottom)
        p = self.param_shape(conv)
        if conv in self.convs:
            if conv in self.spation_convs:
                channels = 1
            else:
                assert s[1]==p[1]
                channels *= p[1]
            outputs *= p[0]
            c = s[2]*s[3]*outputs*channels*p[2]*p[3] / self.conv_param_stride(conv)**2
        elif conv in self.innerproduct:
            c = p[0]*p[1]
        else:
            pass
        return int(c)

    def computation(self, params=False):
        comp=0
        if params:
            NotImplementedError
        else:
            l = []
            for conv in self.convs:
                l.append(self.layercomputation(conv))
        comp = sum(l)
        print("flops", comp)
        for conv,i in zip(self.convs, l):
            print(conv, i, int(i*1000/comp))
        return comp

    def rescomputation(self):
        comp=5036310528
        l = []
        #for conv in self.convs:
        #    l.append(self.layercomputation(conv))
        #comp = sum(l)
        print("flops", comp)
        keep = dcfgs.dic.keep
        newcomp = 0
        for conv in self.convs:
            if 'branch2a' in conv:
                c = self.layercomputation(conv, outputs=keep)#, channels=keep)
            #elif 'branch2b' in conv:
            #    c = self.layercomputation(conv, outputs=keep, channels=keep)
            #elif 'branch2c' in conv:
            #    c = self.layercomputation(conv, channels=keep)
            else:
                c = self.layercomputation(conv)
            print(conv,c)

            newcomp+=c
        print(float(newcomp)/comp)

    def getBNaff(self, bn, affine, scale=1.):
        mean = scale * self.param_data(bn)
        variance = scale * self.param_b_data(bn)**.5
        k = scale * self.param_data(affine)
        b = scale * self.param_b_data(affine)
        return mean, variance, k, b

    def merge_bn(self, DEBUG=0):
        """
        Return:
            merged Weights
        """
        nobias=False
        def scale2tensor(s):
            return s.reshape([len(s), 1, 1, 1])

        BNs = self.type2names("BatchNorm")
        Affines = self.type2names("Scale")
        ReLUs = self.type2names("ReLU")
        Convs = self.type2names()
        assert len(BNs) == len(Affines)

        WPQ = dict()
        for affine in Affines:
            if self.bottom_names[affine][0] in BNs:
                # non inplace BN
                noninplace = True
                bn = self.bottom_names[affine][0]
                conv = self.bottom_names[bn][0]
                assert conv in Convs

            else:
                noninplace = False
                conv = self.bottom_names[affine][0]
                for bn in BNs:
                    if self.bottom_names[bn][0] == conv:
                        break

            triplet = (conv, bn, affine)
            print("Merging", triplet)

            if not DEBUG:
                scale = 1.

                mva = self.param(bn)[2].data[0]
                if mva != scale:
                    raise Exception("Using moving average "+str(mva)+" NotImplemented")
                    #scale /= mva

                mean, variance, k, b = self.getBNaff(bn, affine)
                # y = wx + b
                # (y - mean) / var * k + b
                weights = self.param_data(conv)
                weights = weights / scale2tensor(variance) * scale2tensor(k)

                if len(self.param(conv)) == 1:
                    bias = np.zeros(weights.shape[0])
                    self.set_conv(conv, bias=True)
                    self.param(conv).append(self.param_b(bn))
                    nobias=True
                else:
                    bias = self.param_b_data(conv)
                bias -= mean
                bias = bias / variance * k + b

                WPQ[(conv, 0)] = weights
                WPQ[(conv, 1)] = bias

            self.remove(affine)
            self.remove(bn)
            if not noninplace:
                have_relu=False
                for r in ReLUs:
                    if self.bottom_names[r][0] == conv:
                        have_relu=True
                        break
                if have_relu:
                    self.net_param.ch_top(r, r, conv)
                    for i in self.net_param.layer:
                        if i != r:
                            self.net_param.ch_bottom(i, r, conv)

        if cfgs.mp:
            if not nobias:
                new_pt = self.save_pt(prefix = 'bn')
                return WPQ, new_pt
            else:
                new_pt, new_model = self.save(prefix='bn')
                return WPQ, new_pt, new_model

        new_pt, new_model = self.save(prefix='bn')
        return WPQ, new_pt, new_model

    def invBN(self, arr, Y_name):
        if isinstance(arr, int) or len(self.bns) == 0 or len(self.affines) == 0:
            return arr
        interstellar = Y_name.split('_')[0]
        for i in self.bottom_names[interstellar]:
            if i in self.bns and 'branch2c' in i:
                bn = i
                break
        for i in self.affines:
            if self.layer_bottom(i) == bn:
                affine = i
                break

        if 1: print('inverted bn', bn, affine, Y_name)
        mean, std, k, b = self.getBNaff(bn, affine)
        # (y - mean) / std * k + b
        #return (arr - b) * std / k + mean
        return arr * std / k
        #embed()


    def save_no_bn(self, WPQ, prefix='bn'):
        self.forward()
        for i, j in WPQ.items():
            self.set_param_data(i, j)

        return self.save_caffemodel(prefix=prefix)

    def seperateConvReLU(self, DEBUG=0):
        if dcfgs.model in [cfgs.Models.resnet, cfgs.Models.xception] and len(self.bns) > 1 and len(self.affines) > 1 and dcfgs.res.bn:
            for b,a in zip(self.bns, self.affines):
                if self.top_names[b][0] != b:
                    conv = self.bottom_names[b][0]
                    if conv not in self.convs:
                        continue
                    self.net_param.ch_top(b, b, conv)
                    self.net_param.ch_bottom(a, b, conv)
                    for r in self.relus:
                        if self.top_names[r][0] == conv:
                            self.net_param.ch_bottom(r, b, conv)
                            break
                    for i in self.net_param.layer:
                        if i != b:
                            self.net_param.ch_bottom(i, b, conv)
        else:
            for r in self.relus:
                if self.top_names[r][0] != r:
                    conv = self.bottom_names[r][0]
                    if conv not in self.convs:
                        continue
                    self.net_param.ch_top(r, r, conv)
                    for i in self.net_param.layer:
                        if i != r:
                            self.net_param.ch_bottom(i, r, conv)

        new_pt = self.save_pt(prefix = 'bn')
        return new_pt

    def inlineConvBN(self):
        if dcfgs.res.bn:
            for r,a in zip(self.bns, self.affines):
                if self.top_names[r][0] == r:
                    conv = self.bottom_names[r][0]
                    if conv not in self.convs:
                        continue
                    self.net_param.ch_top(r, conv, r)
                    self.net_param.ch_bottom(a, conv, r)
                    for i in self.relus:
                        if self.top_names[i][0] == r:
                            self.net_param.ch_bottom(i, conv, r)
                            break
                    for i in self.net_param.layer:
                        if i != r:
                            self.net_param.ch_bottom(i, conv, r)
        else:
            pass

    def preprocess_resnet(self):
        # find all shortcuts
        sums = self.type2names('Eltwise')
        convs = self.type2names()
        ReLUs = self.type2names("ReLU")

        projs = {}
        WPQ, pt, model = {}, None, self.caffemodel_dir
        if dcfgs.model not in [cfgs.Models.xception, cfgs.Models.resnet] or not dcfgs.res.bn:
            WPQ, pt, model = self.merge_bn()

        if dcfgs.splitconvrelu:
            pt = self.seperateConvReLU()
        return WPQ, pt, model

    def R3(self): # TODO: Delete VH and ITQ from R3 to eliminate spatial and channel factorization (tried but failed ㅜㅜ) -by Mario
        speed_ratio = dcfgs.dic.keep
        if speed_ratio not in [3.]: # this if-statement might give a problem if we change the speed-up target. Consider adding more values to the list -by Mario
            NotImplementedError
        if dcfgs.dic.vh: # TODO: Consider changing this prefixes to obtained more descriptive names for prototxt files - by Mario
            prefix = '3C'
        else:
            prefix = '2C'
        prefix += str(int(speed_ratio)+1)+'x'
        DEBUG = True
        convs= self.convs
        self.WPQ = dict()
        self.selection = dict()
        self._mem = True
        end = 5 # TODO: Consider passing a flag to create this dictionaries for other models (passign arguments to the paserser maybe?) -by Mario
        alldic = ['conv%d_1' % i for i in range(1,end)] + ['conv%d_2' % i for i in range(3, end)]
        pooldic = ['conv1_2', 'conv2_2']#, 'conv3_3']
        rankdic = {'conv1_1': 17,
                   'conv1_2': 17,
                   'conv2_1': 37,
                   'conv2_2': 47,
                   'conv3_1': 83,
                   'conv3_2': 89,
                   'conv3_3': 106,
                   'conv4_1': 175,
                   'conv4_2': 192,
                   'conv4_3': 227,
                   'conv5_1': 398,
                   'conv5_2': 390,
                   'conv5_3': 379}

        for i in rankdic:
            if 'conv5' in i:
                continue # the break-statemet was giving a bug, so changed it to continue-statement -by Mario
            rankdic[i] = int(rankdic[i] * 4. / speed_ratio)
        c_ratio = 1.15

        def getX(name):
            x = self.extract_XY(self.bottom_names[name][0], name)
            return np.rollaxis(x.reshape((-1, 3, 3, x.shape[1])), 3, 1).copy()

        def setConv(c, d):
            if c in self.selection:
                self.param_data(c)[:,self.selection[c],:,:] = d
            else:
                self.set_param_data(c, d)

        t = Timer()

        for conv, convnext in zip(convs[1:], convs[2:]+['pool5']): # note that we exclude the first conv, conv1_1 contributes little computation -by Mario
            conv_V = underline(conv, 'V')                          # TODO: Consider getting read of this V,H string builders, but keep in mind that
            conv_H = underline(conv, 'H')                          # there is dependency between "channel decomposition" and "channel pruning" -by Mario
            conv_P = underline(conv, 'P')
            W_shape = self.param_shape(conv)
            d_c = int(W_shape[0] / c_ratio)
            rank = rankdic[conv]
            d_prime = rank
            if d_c < rank: d_c = rank
            '''spatial decomposition'''
            if True:
                t.tic()
                weights = self.param_data(conv)
                if conv in self.selection:
                    weights = weights[:,self.selection[conv],:,:]
                if 1:
                    Y = self._feats_dict[conv] - self.param_b_data(conv)
                    X = getX(conv)
                    if conv in self.selection:
                        X = X[:,self.selection[conv],:,:]
                    V, H, VHr, b = VH_decompose(weights, rank=rank, DEBUG=DEBUG, X=X, Y=Y)
                    self.set_param_b(conv,b)
                else:
                    V, H, VHr = VH_decompose(weights, rank=rank, DEBUG=DEBUG)


                self.WPQ[conv_V] = V

                # set W to low rank W, asymetric solver
                setConv(conv,VHr)

                self.WPQ[(conv_H, 0)] = H
                self.WPQ[(conv_H, 1)] = self.param_b_data(conv)
                if 0:#DEBUG:
                    print("W", W_shape)
                    print("V", V.shape)
                    print("H", H.shape)

                t.toc('spatial_decomposition')

            self.insert(conv, conv_H)

            '''channel decomposition'''
            if True:# and conv != 'conv3_3':
                t.tic()
                feats_dict, _ = self.extract_features(names=conv, points_dict=self._points_dict, save=1)
                Y = feats_dict[conv]
                W1, W2, B, W12 = ITQ_decompose(Y, self._feats_dict[conv], H, d_prime, bias=self.param_b_data(conv), DEBUG=0, Wr=VHr)

                # set W to low rank W, asymetric solver
                setConv(conv,W12.copy())
                self.set_param_b(conv, B.copy())

                # save W_prime and P params
                W_prime_shape = [d_prime, H.shape[1], H.shape[2], H.shape[3]]
                P_shape = [W2.shape[0], W2.shape[1], 1, 1]
                self.WPQ[(conv_H, 0)] = W1.reshape(W_prime_shape)
                self.WPQ[(conv_H, 1)] = np.zeros(d_prime)
                self.WPQ[(conv_P, 0)] = W2.reshape(P_shape)
                self.WPQ[(conv_P, 1)] = B

                self.insert(conv_H, conv_P, pad=0, kernel_size=1, bias=True, stride=1)

                t.toc('channel_decomposition')

            '''channel pruning'''
            if dcfgs.dic.vh and (conv in alldic or conv in pooldic) and (convnext in self.convs):
                t.tic()
                #if conv.startswith('conv4'): #what is this -by Mario
                #    c_ratio = 1.5
                if conv in pooldic:
                    X_name = self.bottom_names[convnext][0]
                else:
                    X_name = conv
                """
                dictionary_kernel() is a function wrapper of the fucntion dictionary() which
                performes the feature maps selection (lasso reression and mean square error minization)

                ===== Relation to Figure 2 of the paper =====
                - layer B = conv = Xname
                  The conv layer from which we will remove feature maps

                  W1! shape: (c_outs,c_in, k_h, k_w)

                  Weiths from this layer are denoted as W1. Filters in W1
                  are removed at the very end (simply use the index of the removed
                  feature maps of this layer to remove the corresponding filters)

                - layer C = convnext = Y_name
                  The conv layer  that we use as ground truth to minimize the
                  error during B's feature map selection, that is, theb squared error of the sampled points
                  stored in feats_dict ) and the real feature activations.

                  W2! (c_outs,c_in, k_h, k_w)

                  Weighs of this layer are denoted as W2. The value of these weights is used during
                  the lasso regression execution
                  -by Mario
                """

                #idxs, array of booleans that indicates which feature maps(?) are elimated
                # newX: N c h w (BatchSize, channels, h, w), W2: n c h w (out_channels, in_channels, fitler_h, filter_w)
                idxs, W2, B2 = self.dictionary_kernel(X_name, None, d_c, convnext, None)
                # W2
                self.selection[convnext] = idxs
                self.param_data(convnext)[:, ~idxs, ...] = 0
                self.param_data(convnext)[:, idxs, ...] = W2.copy()
                self.set_param_b(convnext,B2)
                # W1 #TODO: For channel pruning only, we should handle the origial conv layers (not the _H or _P layers)
                     # This section of code must be addapted
                if (conv_P,0) in self.WPQ:
                    key =  conv_P
                else:
                    key = conv_H
                self.WPQ[(key,0)] = self.WPQ[(key,0)][idxs]
                self.WPQ[(key,1)] = self.WPQ[(key,1)][idxs]
                self.set_conv(key, num_output=sum(idxs))

                t.toc('channel_pruning')
            # setup V
            H_params = {'bias':True}
            H_params.update(self.infer_pad_kernel(self.WPQ[(conv_H, 0)], conv))
            self.set_conv(conv_H, **H_params)
            # setup H
            V_params = self.infer_pad_kernel(self.WPQ[conv_V], conv)
            self.set_conv(conv, new_name=conv_V, **V_params)
            if 0:#DEBUG:
                print("V", H_params)
                print("H", V_params)
        new_pt = self.save_pt(prefix=prefix)
        return self.WPQ, new_pt

    def combineHP(self):
        H = []
        P = []
        for conv in self.convs:
            if conv.endswith('_H'):
                H.append(conv)
                continue
            if conv.endswith('_P'):
                P.append(conv)
                continue
        assert len(H) == len(P)
        for h,p in zip(H,P):
            assert h.split('_H')[0] ==  p.split('_P')[0]
            Hshape = self.param_shape(h)
            m  =Hshape[0]
            o = self.param_shape(p)[0]
            if 3 * m >= 2 * o:
                newshape = list(Hshape)
                newshape[0] = o
                Hw = self.param_data(h).reshape((m,-1))
                Pw = self.param_data(p).reshape((o,-1))
                Hb = self.param_b_data(h)
                pb = self.param_b_data(p)
                neww = Pw.dot(Hw).reshape(newshape)
                newb = pb + Pw.dot(Hb)

                self.ch_param_data(h, neww)
                self.ch_param_b(h, newb)
                self.set_conv(h, num_output=o)
                self.remove(p)
        a,b = self.save(prefix='cb')
        print('-model',a,'-weights',b)



    def autodet_rank(self, layers, speedratio):
        NotImplementedError
        return ranks

    def asymmetric3D(self):
        pass

    def findtopby_type(self, bns, conv):
        for bn in bns:
            if conv in self.bottom_names[bn]:
                return bn
        return None

    def W1keep(self, conv, idxs):
        if dcfgs.model == cfgs.Models.xception:
            assert conv in self.spation_convs
            sconv = conv

            assert len(self.bottom_names[sconv]) == 1
            conv = self.layer_bottom(conv) # self.bottom_names[self.layer_bottom(sconv)][0]

            if (sconv,0) in self.WPQ:
                self.WPQ[(sconv,0)] = self.WPQ[(sconv,0)][idxs]
            else:
                self.WPQ[(sconv,0)] = self.param_data(sconv)[idxs].copy()
            self.param_data(sconv)[~idxs, ...] = 0.
            o=len(self.WPQ[(sconv,0)])
            self.set_conv(sconv, num_output=o, group=o)

        if conv in (self.sums + self.pools):
            if dcfgs.model in [cfgs.Models.xception]:
                self.bottoms2ch.append([conv, sconv, idxs])
                return
            if dcfgs.model in [cfgs.Models.resnet]:
                for i in self.convs:
                    if self.layer_bottom(i) == conv:
                        sconv = i
                self.bottoms2ch.append([conv, sconv, idxs])
                return
            elif dcfgs.model == cfgs.Models.vgg:
                conv = self.layer_bottom(conv)

        bn = None
        affine = None
        if conv in self.bns:
            bn = conv
            for i in self.affines:
                if self.layer_bottom(i) == bn:
                    affine = i
                    break
        #if (dcfgs.kernelname != cfgs.kernels.pruning) and dcfgs.model not in [cfgs.Models.resnet,cfgs.Models.rescifar]:
        if conv not in self.convs:
            conv = self.bottom_names[conv][0]
        else:
            for i in self.affines:
                if self.layer_bottom(i) == conv:
                    affine = i
                    break
            for i in self.bns:
                if self.layer_bottom(i) == conv:
                    bn = i
                    break
        #self.bnidx.append([conv, idxs])
        W1 = self.param_data(conv)[idxs,...]
        b = self.param_b_data(conv)[idxs]
        if (conv,0) in self.WPQ:
            self.WPQ[(conv, 0)] = self.WPQ[(conv,0)][idxs, ...].copy()
        else:
            self.WPQ[(conv, 0)] = W1.copy()
        if (conv,1) in self.WPQ:
            self.WPQ[(conv, 1)] = self.WPQ[(conv,1)][idxs].copy()
        else:
            self.WPQ[(conv, 1)] = b.copy()
        if bn is not None:
            self.WPQ[(bn, 0)] =self.param_data(bn)[idxs].copy()
            self.WPQ[(bn, 1)] = self.param_b_data(bn)[idxs].copy()
        if affine is not None:
            self.WPQ[(affine, 0)] = self.param_data(affine)[idxs].copy()
            self.WPQ[(affine, 1)] = self.param_b_data(affine)[idxs].copy()
        if 1:
            self.param_data(conv)[~idxs, ...] = 0.
            self.param_data(conv)[idxs, ...] = W1
            #self.ch_param_data(conv, W1)
            #self.ch_param_b(conv, b[idxs])
            self.param_b_data(conv)[~idxs] = 0.
            self.param_b_data(conv)[idxs] = b
            if bn is not None:
                #self.ch_param_data(bn, self.param_data(bn)[idxs])
                self.param_data(bn)[~idxs] = 0.
                #self.param_data(bn)[idxs] = self.param_data(bn)[idxs]
                #self.ch_param_b(bn, self.param_b_data(bn)[idxs])
                self.param_b_data(bn)[~idxs] = 0.
                #self.param_b_data(bn)[idxs] = self.param_b_data(bn)[idxs]
            if affine is not None:
                #self.ch_param_data(affine, self.param_data(affine)[idxs])
                self.param_data(affine)[~idxs] = 0.
                #self.param_data(affine)[idxs] = self.param_data(affine)[idxs]
                #self.ch_param_b(affine, self.param_b_data(affine)[idxs])
                self.param_b_data(affine)[~idxs] = 0.
                #self.param_b_data(affine)[idxs] = self.param_b_data(affine)[idxs]
        self.set_conv(conv, num_output=len(self.WPQ[(conv, 1)]))

    def W2keep(self, top, idxs, W2, B2=None, layerbylayer=False):
        asym=1
        #W2 = self.param_data(top)[:,idxs,:,:]
        if asym:
            self.param_data(top)[:, ~idxs, ...] = 0
            self.param_data(top)[:, idxs, ...] = W2.copy()
        #self.ch_param_data(top, W2)
        self.WPQ[(top, 0)] = W2.copy()
        if B2 is not None:
            newB2 = B2.copy()
            if (not layerbylayer) and (dcfgs.ls != cfgs.solvers.gd or not self._mem): # add paramb
                newB2 += self.param_b_data(top)
            if asym:
                self.set_param_b(top, newB2)
            self.WPQ[(top, 1)] = newB2.copy()
            #self.set_param_b(top, newB2)

    def select(self, name, nextname, idxs):
        fname = self.net_param.selector(name, nextname, self.blobs_shape(name), int(sum(idxs)))
        self.nonWPQ[fname] = idxs.astype(int)
        if 0: print(self.nonWPQ[fname])

    def pruning_kernel(self, X, weights, d_prime, W2_name, Y):
        idxs = np.argsort(-np.abs(weights).sum((1,2,3)))
        idxs = np.sort(idxs[:d_prime])
        W2 = self.param_data(W2_name)[:,idxs,:,:]
        if 1: print(W2_name)
        newidxs = np.zeros(len(weights)).astype(bool)
        newidxs[idxs] = True
        return newidxs, W2, None

    def appresb(self,Y_name):
        residual_B = 0
        def extractResB(a):
            nonlocal residual_B
            feats_dict, _ = self.extract_features([a], points_dict=self._points_dict,save=1)
            residual_B = (self._feats_dict[a]) - (feats_dict[a])

        if dcfgs.res.short == 1:
            if dcfgs.dic.option==cfgs.pruning_options.resnet:
                b2c = '_branch2c'
                if b2c in Y_name:
                    b1sum = Y_name.partition(b2c)[0]
                    # assert X_name.startswith(b1sum)
                    if b1sum +'_branch1' in self.convs:
                        if len(self.bns) == 0:
                            extractResB(b1sum +'_branch1')
                        else:
                            for bn in self.bottom_names[b1sum]:
                                if bn in self.bns and 'branch1' in bn:
                                    extractResB(bn)
                                    print("extracted", bn)
                                    for k in range(dcfgs.nBatches):
                                        for kk, jj in zip(self._points_dict[(k, Y_name , "randx")] , self._points_dict[(k, bn , "randx")]):
                                            assert kk==jj
                                    break
                    else:
                        b1sum = self.sums[self.sums.index(b1sum) - 1]
                        extractResB(b1sum)
            elif dcfgs.dic.option==1:
                if DEBUG: redprint("This line executed because dcfgs.dic.option is set to 1 [net.appresb()]")
                b2c = '_conv1'
                if Y_name.endswith(b2c):
                    fsums = ['first_conv'] + self.sums
                    blockname = Y_name.partition(b2c)[0]
                    blockproj = blockname + '_proj'
                    b1sum = fsums[fsums.index(blockname+'_sum')-1]
                    if blockproj in self.convs:
                        b1sum = blockproj
                    extractResB(b1sum)

        if not isinstance(residual_B, int):
            print("apprixiamating A + resB", dcfgs.dic.option)
        return residual_B

    def dictionary_kernel(self, X_name, weights, d_prime, Y_name, Y, DEBUG = 0):
        """ channel pruning algorithm wrapper
        X_name: the conv layer to prune
        weights: deprecated
        d_prime: number of preserving channels (c' in paper), the speed-up ratio = d_prime / number of channels
        Y_name: the next conv layer (For later removing of corresponding pruned weights)
        Y: deprecated
        """
        # weights,Y is None
        if not self._mem: # if we have not extracted sample features before, then extract them
            feats_dict, points_dict = self.extract_features([X_name, Y_name], save=1)
            self.load_frozen(feats_dict=feats_dict, points_dict=points_dict )

        X = self.extract_XY(X_name, Y_name) # extract_XY(conv, convnext)
        N = self.blobs_num(Y_name)
        h = self.param_shape(Y_name)[-1]
        w=h
        newX = np.rollaxis(X.reshape((-1, h, w, X.shape[1])), 3, 1).copy()

        W2 = self.param_data(Y_name)
        if dcfgs.ls != cfgs.solvers.gd or not self._mem: # add paramb # what is this? -by Mario
            if DEBUG: print("net.dictionary_kernel: dcfgs.ls is not gd or there is no MemoryData -by Mario")
            gtY = self._feats_dict[Y_name] - self.param_b_data(Y_name) # compute the difference between what the extracted feature and the biases of the next layer ??? -by Mario
            if 0:
                print("warning, not accumulative")
                feats_dict, _ = self.extract_features([Y_name],points_dict=self._points_dict, save=1)
                Y = feats_dict[Y_name] - self.param_b_data(Y_name)
            else:
                if DEBUG: print("gtY is only defined in this if-condition, but it is required bellow --> this condition is always be true?")
                Y = gtY

            resY = self.appresb(Y_name)
            if dcfgs.model in [cfgs.Models.xception, cfgs.Models.resnet]:
                resY = self.invBN(resY, Y_name)
            else:
                newX = relu(newX)
            #if resY != 0: embed()
            Y+= resY
        else:
            Y=newX.reshape(newX.shape[0],-1).dot(W2.reshape((W2.shape[0],-1)).T)

        print("rMSE", rel_error(newX.reshape((newX.shape[0],-1)).dot(W2.reshape((W2.shape[0],-1)).T), gtY))
        # performe the lasso regression -by Mario
        outputs = dictionary(newX, W2, Y, rank=d_prime, B2=self.param_b_data(Y_name))
        if dcfgs.ls == cfgs.solvers.gd: self.caffe_device() # if the solver is gd(what is gd?), set GPU operation
        #Y_shape = self.param_shape(Y_name)
        #X_shape = self.param_shape(X_name)
        #self.prunedweights+= (Y_shape[1] - len(np.where(outputs[0])[0])) * (Y_shape[0]*Y_shape[2]*Y_shape[3] + X_shape[1]*X_shape[2]*X_shape[3])
        #print("pruned", self.prunedweights)
        # newX: N c h w,  W2: n c h w
        return outputs

def load_layer(convs, frozen_layer):
    _points_dict = dict()
    _feats_dict = dict()

    def subfile(filename):
        return osp.join(frozen_layer, filename)

    with open(subfile(cfgs._points_dict_name), 'rb') as f:
        _points_dict = pickle.load(f)
    for conv in convs:
        filename = subfile(conv)
        if osp.exists(filename):
            print("loading ", conv)
            with open(filename, 'rb') as f:
                _feats_dict[conv] = pickle.load(f)

    return _points_dict, _feats_dict

def caffe_test(pt, model, gpu=None, time=True):
    if gpu is None:
        gpu = cfgs.caffe_vis
    shell(cfgs.caffe_path, 'test', '-gpu', gpu, '-weights', model, '-model', pt)
    if time:
        shell(cfgs.caffe_path, 'time', '-gpu', gpu, '-weights', model, '-model', pt)
