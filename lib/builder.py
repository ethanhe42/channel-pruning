from __future__ import print_function
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import os.path as osp
import os
import sys
from collections import OrderedDict
from .utils import underline
# import caffe
from IPython import embed

debug=True

class Solver:
    p = caffe_pb2.SolverParameter()

    class policy:
        """    - fixed: always return base_lr."""
        fixed = 'fixed'
        """    - step: return base_lr * gamma ^ (floor(iter / step))"""
        step = 'step'
        """    - exp: return base_lr * gamma ^ iter"""
        exp = 'exp'
        """    - inv: return base_lr * (1 + gamma * iter) ^ (- power)"""
        inv = 'inv'
        """    - multistep: similar to step but it allows non uniform steps defined by stepvalue"""
        multistep = 'multistep'
        """    - poly: the effective learning rate follows a polynomial decay, to be zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)"""
        poly = 'poly'
        """    - sigmoid: the effective learning rate follows a sigmod decay return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))"""
        sigmoid = 'sigmoid'

    class method:
        nesterov = "Nesterov"
        SGD = "SGD"
        AdaGrad = "AdaGrad"
        RMSProp = "RMSProp"
        AdaDelta = "AdaDelta"
        Adam = "Adam"


    def __init__(self, solver_name=None, folder=None, lr_policy=None):

        class Machine:
            GPU = self.p.GPU
            CPU = self.p.GPU        
        self.machine = Machine()
        
        if solver_name is not None:
            filepath, ext = osp.splitext(solver_name)
            if ext == '':
                ext = '.prototxt'
                self.name = filepath+ext
            else:
                self.name = solver_name
        else:
            self.name = 'solver.prototxt'

        self.folder = folder
        
        if self.folder is not None:
            if not osp.exists(self.folder):
                os.mkdir(self.folder)
            self.name = osp.join(self.folder, self.name) 

        # defaults

        self.p.test_iter.extend([100])
        self.p.test_interval = 1000
        self.p.test_initialization = True

        self.p.base_lr = 0.1

        if lr_policy is not None:
            self.p.lr_policy = lr_policy
        else:
            self.p.lr_policy = self.policy.multistep

        if self.p.lr_policy == self.policy.multistep:
            self.p.stepvalue.extend([32000, 48000])
        if self.p.lr_policy not in [self.policy.fixed, self.policy.sigmoid]:
            self.p.gamma = 0.1

        self.p.momentum = 0.9
        self.p.weight_decay = 0.0001

        self.p.display = 100
        self.p.max_iter = 64000
        self.p.snapshot = 10000
        self.p.snapshot_prefix = osp.join(self.folder, "snapshot/")
        self.logs = osp.join(self.folder, "logs/")
        self.p.solver_mode = self.machine.GPU

        self.p.type = self.method.nesterov
        self.p.net = osp.join(self.folder, "trainval.prototxt")
    
    def set_lr(self, lr):
        self.p.base_lr = lr
    
    def set_max_iter(self, max_iter):
        self.p.max_iter = max_iter
        if max_iter < self.p.snapshot:
            self.p.snapshot = max_iter
        if max_iter < self.p.test_interval:
            self.p.test_interval = max_iter
    
    def write(self):
        # dirname = osp.dirname(name)
        # if not osp.exists(dirname):
        #     os.mkdir(dirname)
        for i in [self.p.snapshot_prefix, self.logs]:
            if not osp.exists(i):
                os.mkdir(i)
        with open(self.name, 'wb') as f:
            f.write(str(self.p))
        return self.name

class Net:
    class filler:
        msra = "msra"
        xavier = "xavier"
        orthogonal = "orthogonal"
        constant = "constant"

    def __init__(self, name="network", pt=None):
        self.net = caffe_pb2.NetParameter()
        if pt is None:
            self.net.name = name
        else:
            with open(pt, 'rt') as f:
                pb2.text_format.Merge(f.read(), self.net)
        self.bottom = None
        self.cur = None
        self.this = None

        self._layer = None
        self._bottom = None
    
    @property
    def layer(self):
        """dict of list of ptrs"""
        # TODO caching invoke some potential bugs
        # if self._layer is None:
        self._layer = OrderedDict()
        for layer in self.net.layer:
            if layer.name not in self._layer:
                self._layer[layer.name] = []
            self._layer[layer.name] += [layer]
        return self._layer

    def set_cur(self, bottom):
        self.cur = self.layer[bottom][0]

    def layer_type(self, layer_name):
        return self.layer[layer_name][0].type

    def layer_bottom(self, layer_name):
        bottom = self.layer[layer_name][0].bottom
        if len(bottom) == 1:
            return bottom[0]
        return bottom
    
    def set_bottom(self, layer_name, newbottom):
        for i in self.layer[layer_name]:
            # TODO add multi bottom support 
            # i.bottom.extend([newbottom] if isinstance(newbottom, str) else newbottom)
            i.bottom.extend([newbottom])
    
    def set_top(self, layer_name, newtop):
        for i in self.layer[layer_name]:
            # TODO add multi top support 
            # i.top.extend([newtop] if isinstance(newtop, str) else newtop)
            i.top.extend([newtop])

    def rm_bottom(self, layer_name, bottom):
        for i in self.layer[layer_name]:
            if bottom in i.bottom:
                i.bottom.remove(bottom)
            else:
                return False
        return True

    def rm_top(self, layer_name, top):
        for i in self.layer[layer_name]:
            if top in i.top:
                i.top.remove(top)
            else:
                return False
        return True

    def ch_bottom(self, layer_name, newbottom, oldbottom):
        '''
        if inline layer, also change top
        '''
        if not self.rm_bottom(layer_name, oldbottom):
            return False
        self.set_bottom(layer_name, newbottom)
        if self.layer_top(layer_name) == oldbottom:
            if 0: print("should use seperateReLU", layer_name)
            self.ch_top(layer_name, newbottom, oldbottom)
        return True

    def ch_top(self, layer_name, newtop, oldtop):
        if not self.rm_top(layer_name, oldtop):
            return False
        self.set_top(layer_name, newtop)
        return True
    
    def set_name(self, layer_name, newname):
        for i in self.layer[layer_name]:
            i.name = newname
            if layer_name in i.top:
                i.top.remove(layer_name)
                i.top.extend([newname])

    def ch_name(self, name, newname):
        self.set_name(name, newname)
        for i in self.layer:
            self.ch_bottom(i, newname, name)

    def bringforward(self, bottomname, afterinline=False):
        newlayer = self.net.layer[-1]
        self.net.layer.remove(newlayer)
        appendix = []
        
        if bottomname is None:
            # bring forward to first
            while len(self.net.layer):
                behind = self.net.layer[-1]
                appendix.append(behind)
                self.net.layer.remove(behind)                
        else:
            tmp = self.net.layer[-1].name
            while tmp != bottomname:
                if afterinline:
                    try:
                        if self.layer_top(tmp) == bottomname:
                            break
                    except:
                        pass
                behind = self.net.layer[-1]
                appendix.append(behind)
                self.net.layer.remove(behind)
                tmp = self.net.layer[-1].name

        appendix.append(newlayer)
        self.net.layer.extend(appendix[::-1])
    
    def rm_layer(self, name, inplace=False):
        if not inplace:
            bottom = self.layer[name][0].bottom
            assert len(bottom) == 1
            bottom = bottom[0]

        for i in self.layer[name]:
            self.net.layer.remove(i)
            
        if not inplace:
            for i in self.layer:
                self.ch_bottom(i, bottom, name)
                #if self.ch_bottom(i, bottom, name):
                #    for j in self.layer:
                #        self.ch_top(j, i, name)
                #        self.ch_bottom(j, i, name)
                # self.ch_top(i, bottom, name)

    def split(self, bottom, top):
        for i in self.layer:
            self.ch_top(i, i, bottom)
            self.ch_bottom(i, top, bottom)
                
    # def ch_name(self, layer_name, newname):
    #     for i in self.layer[layer_name]:
    #         i.name = newname
    #         if layer_name in i.top:
    #             i.top.remove(layer_name)
    #             i.top.extend([newname])
    #             for j in self.layer:
    #                 if layer_name in self.layer[j].bottom:
    #                     self.ch_bottom(j, newname, )


    def layer_top(self, layer_name):
        top = self.layer[layer_name][0].top
        if len(top) == 1:
            return top[0]
        return top

    def setup(self, name, layer_type, bottom=[], top=[], inplace=False):
        self.bottom = self.cur

        new_layer = self.net.layer.add()

        new_layer.name = name
        new_layer.type = layer_type

        if self.bottom is not None and new_layer.type not in ['DummyData', 'Data']:
            bottom_name = [self.bottom.name]
            if len(bottom) == 0:
                bottom = bottom_name
            new_layer.bottom.extend(bottom)
        
        if inplace:
            if len(top) == 0:
                top = bottom_name
        elif len(top) == 0:
            top = [name]
        new_layer.top.extend(top)

        self.this = new_layer
        if not inplace:
            self.cur = new_layer

    def suffix(self, name, self_name=None):
        if self_name is None:
            return self.cur.name + '_' + name
        else:
            return self_name

    def write(self, name=None, folder=None):
        # dirname = osp.dirname(name)
        # if not osp.exists(dirname):
        #     os.mkdir(dirname)
        if folder is not None:
            name = osp.join(folder, 'trainval.prototxt')
        elif name is None:
            name = 'trainval.pt'
        else:
            filepath, ext = osp.splitext(name)
            if ext == '':
                ext = '.prototxt'
                name = filepath+ext

        with open(name, 'w') as f:
            f.write(str(self.net))
        return name

    def show(self):
        print(self.net)

    def type2names(self, layer_type):
        names = []
        for i in self.layer:
            if self.layer_type(i) == layer_type:
                names.append(i)

        return names

    #************************** params **************************

    def param(self, lr_mult=1, decay_mult=0, name=None):
        new_param = self.this.param.add()
        if name is not None: # for sharing weights
            new_param.name = name
        new_param.lr_mult = lr_mult
        new_param.decay_mult = decay_mult

    def transform_param(self, mean_value=128, batch_size=128, scale=.0078125, mirror=1, crop_size=None, mean_file_size=None, phase=None):

        new_transform_param = self.this.transform_param
        if scale != 1:
            new_transform_param.scale = scale
        if isinstance(mean_value, list):
            new_transform_param.mean_value.extend(mean_value)
        else:
            new_transform_param.mean_value.extend([mean_value])
        if phase is not None and phase == 'TEST':
            return

        new_transform_param.mirror = mirror
        if crop_size is not None:
            new_transform_param.crop_size = crop_size
        

    def data_param(self, source, backend='LMDB', batch_size=128):
        new_data_param = self.this.data_param
        new_data_param.source = source
        if backend == 'LMDB':
            new_data_param.backend = new_data_param.LMDB
        else:
            NotImplementedError
        new_data_param.batch_size = batch_size    

    def weight_filler(self, filler=filler.msra, std=None):
        """xavier"""
        if self.this.type == 'InnerProduct':
            w_filler = self.this.inner_product_param.weight_filler
        else:
            w_filler = self.this.convolution_param.weight_filler
        w_filler.type = filler
        if filler == self.filler.orthogonal:
            if std is not None:
                NotImplementedError
                w_filler.std = std

    
    def bias_filler(self, filler='constant', value=0):
        if self.this.type == 'InnerProduct':
            self.this.inner_product_param.bias_filler.type = filler
            self.this.inner_product_param.bias_filler.value = value
        else:
            self.this.convolution_param.bias_filler.type = filler
            self.this.convolution_param.bias_filler.value = value

    def include(self, phase='TRAIN'):
        if phase is not None:
            includes = self.this.include.add()
            if phase == 'TRAIN':
                includes.phase = caffe_pb2.TRAIN
            elif phase == 'TEST':
                includes.phase = caffe_pb2.TEST
        else:
            NotImplementedError


    #************************** inplace **************************
    def ReLU(self, name=None, inplace=True):
        self.setup(self.suffix('relu', name), 'ReLU', inplace=inplace)
    
    def BatchNorm(self, name=None, inplace=True,eps=1e-5):
        moving_average_fraction = 0
        if not inplace:
            bottom = self.this.name
        # train
        bn_name = self.suffix('bn', name)
        self.setup(bn_name, 'BatchNorm', inplace=inplace)
        # self.include()

        self.param(lr_mult=0, decay_mult=0)
        self.param(lr_mult=0, decay_mult=0)
        self.param(lr_mult=0, decay_mult=0)
        batch_norm_param = self.this.batch_norm_param
        if eps != 1e-5:
            batch_norm_param.eps = eps

        return bn_name
        # batch_norm_param.use_global_stats = False
        #batch_norm_param.moving_average_fraction = moving_average_fraction

        # test 
        # if not inplace:
        #     self.setup(bn_name, 'BatchNorm', inplace=inplace, bottom=[bottom])
        # else:
        #     self.setup(bn_name, 'BatchNorm', inplace=inplace)

        # self.include(phase='TEST')

        # self.param(lr_mult=0, decay_mult=0)
        # self.param(lr_mult=0, decay_mult=0)
        # self.param(lr_mult=0, decay_mult=0)
        # batch_norm_param = self.this.batch_norm_param
        # batch_norm_param.use_global_stats = True
        # batch_norm_param.moving_average_fraction = moving_average_fraction

    def Scale(self, name=None, inplace=True, const=False, suf=True):
        sname = self.suffix('scale', name)
        self.setup(sname, 'Scale', inplace=inplace)
        if const:
            self.param(0,0)
            self.param(0,0)
            NotImplementedError
        else:
            self.this.scale_param.bias_term = True
        return sname

    #************************** layers **************************

    def Data(self, source, top=['data', 'label'], name="data", phase=None, **kwargs):
        self.setup(name, 'Data', top=top)

        self.include(phase)

        self.data_param(source)
        self.transform_param(phase=phase, **kwargs)

    def MemoryData(self, batch_size, channels, height, width, top=['data', 'label'], name="data"):
        self.setup(name, 'MemoryData', top=top)

        self.this.memory_data_param.batch_size = batch_size
        self.this.memory_data_param.channels = channels
        self.this.memory_data_param.height = height
        self.this.memory_data_param.width = width
        
    def Convolution(self, 
                    name, 
                    bottom=[], 
                    num_output=None, 
                    kernel_size=3, 
                    pad=1, 
                    stride=1, 
                    decay = True, 
                    bias = False, 
                    freeze = False,
                    filler = filler.msra,
                    phase=None,
                    group=1,
                    engine=2):
        self.setup(name, 'Convolution', bottom=bottom, top=[name])
        
        conv_param = self.this.convolution_param
        if num_output is None:
            num_output = self.bottom.convolution_param.num_output

        conv_param.num_output = num_output
        conv_param.pad.extend([pad])
        conv_param.kernel_size.extend([kernel_size])
        if engine != 2:
            conv_param.engine = engine
        if group != 1:
            conv_param.group=group
        if isinstance(stride, list):
            conv_param.stride.extend(stride)
        else:
            conv_param.stride.extend([stride])
        
        if freeze:
            lr_mult = 0
        else:
            lr_mult = 1
        if decay:
            decay_mult = 1
        else:
            decay_mult = 0
        self.param(lr_mult=lr_mult, decay_mult=decay_mult)
        self.weight_filler(filler)

        if bias:
            if decay:
                decay_mult = 2
            else:
                decay_mult = 0
            self.param(lr_mult=lr_mult, decay_mult=decay_mult)
            self.bias_filler()
        else:
            conv_param.bias_term = False

        if phase is not None:
                self.include(phase)
        
    def SoftmaxWithLoss(self, name='loss', label='label'):
        self.setup(name, 'SoftmaxWithLoss', bottom=[self.cur.name, label])

    def EuclideanLoss(self, name, bottom, loss_weight=1, **kwargs):
        self.setup(name, 'EuclideanLoss', bottom=bottom, top=[name], inplace=True)
        self.this.loss_weight.extend([loss_weight])
    
        for val_name, val in kwargs.items():
            if val_name=='phase':
                self.include(val)
        

    def Softmax(self,bottom=[], name='softmax'):
        self.setup(name, 'Softmax', bottom=bottom)

    def Accuracy(self, name='Accuracy', label='label'):
        self.setup(name, 'Accuracy', bottom=[self.cur.name, label])


    def InnerProduct(self, name='fc', num_output=10, filler=filler.msra):
        self.setup(name, 'InnerProduct')
        self.param(lr_mult=1, decay_mult=1)
        self.param(lr_mult=2, decay_mult=0)    
        inner_product_param = self.this.inner_product_param
        inner_product_param.num_output = num_output
        self.weight_filler(filler)
        self.bias_filler()
    
    def Pooling(self, name, pool='AVE', global_pooling=False, pad=0,stride=1, kernel_size=3):
        """MAX AVE """
        self.setup(name,'Pooling')
        if pool == 'AVE':
            self.this.pooling_param.pool = self.this.pooling_param.AVE
        else:
            NotImplementedError
        if pad != 0:
            self.this.pooling_param.pad = pad
        if stride != 1:
            self.this.pooling_param.stride = stride       
        if not global_pooling:
            self.this.pooling_param.kernel_size = kernel_size       
        self.this.pooling_param.global_pooling = global_pooling

    def Eltwise(self, name, bottom1, operation='SUM', bottom0=None):
        if bottom0 is None:
            bottom0 = self.bottom.name
        self.setup(name, 'Eltwise', bottom=[bottom0, bottom1])
        if operation == 'SUM':
            self.this.eltwise_param.operation = self.this.eltwise_param.SUM
        else:
            NotImplementedError

    def Dropout(self, name=None, inplace=True, ratio=0.5):
        self.setup(self.suffix('dropout', name), 'Dropout', inplace=inplace)
        self.this.dropout_param.dropout_ratio = ratio
    
    def Python(self, module, layer, loss_weight=0, bottom=[],top=[], name='Python', **kwargs):
        self.setup(name, 'Python', bottom=bottom, top=top, inplace=True)
        python_param = self.this.python_param
        python_param.module = module
        python_param.layer = layer
        for val_name, val in kwargs.items():
            if val_name=='phase':
                self.include(val)
        if loss_weight !=0:
            self.this.loss_weight.extend([loss_weight])

    def MVN(self, name=None, bottom=[], normalize_variance=True, across_channels=False, phase='TRAIN'):
        if across_channels:
            NotImplementedError
        if not normalize_variance:
            NotImplementedError
        self.setup(self.suffix('MVN', name),bottom=bottom, layer_type='MVN')
        if phase!='TRAIN':
            NotImplementedError
        self.include()

    def Flatten(self, axis=1, name=None, bottom=None):
        if bottom is not None:
            self.setup(self.suffix('flatten',name), 'Flatten', bottom=[bottom])
        else:
            self.setup(self.suffix('flatten',name), 'Flatten')
        if axis !=  1:
            self.this.flatten_param.axis=axis
        self.include()
        return self.this.name

    def Slice(self, slice_point, bottom, axis=1, phase=None):
        self.setup(underline('slice',bottom), 'Slice', bottom=[bottom])
        if axis !=  1:
            self.this.slice_param.axis=axis

        if phase is not None:
            self.include(phase)

        while True:
            if len(self.this.top):
                self.this.top.pop()
            else:
                break
        
        a = underline('a',bottom)
        b = underline('b',bottom)
        self.this.top.extend([a, b])
        if not isinstance(slice_point, list):
            slice_point = [slice_point]
        self.this.slice_param.slice_point.extend(slice_point)
        return self.this.name, a,b

    def Transpose(self):
        pass

    def DummyData(self, shape, name=None):
        dummyname = self.suffix('DummyData',name)
        self.setup(dummyname, 'DummyData')
        self.this.dummy_data_param.shape.extend(shape)
        return dummyname

    def Filter(self, bottom0, name=None):
        filtername = self.suffix('Filter',name)
        self.setup(filtername, 'Filter', bottom=[bottom0])
        self.param(lr_mult=0, decay_mult=0)
        # self.this.filter_param.weight_filler.type = self.filler.constant
        return filtername

    def selector(self, bottom, top, shape, num_output):
        self.set_cur(bottom)
        fname = self.Filter(bottom)
        self.this.filter_param.num_output = num_output
        self.bringforward(bottom, afterinline=True)
        self.ch_bottom(top, fname, bottom)
        return fname

    def Reduction(self, axis=0, operation='MEAN', name=None, bottom=None):
        if bottom is not None:
            self.setup(self.suffix('Reduction',name), 'Reduction', bottom=[bottom])
        else:
            self.setup(self.suffix('Reduction',name), 'Reduction')

        if axis !=  0:
            self.this.reduction_param.axis=axis
        self.operation = operation
        return self.this.name

    #************************** DIY **************************
    def conv_relu(self, name, relu_name=None, **kwargs):
        self.Convolution(name, **kwargs)
        self.ReLU(relu_name)

    def conv_bn_relu(self, name, bn_name=None, relu_name=None, **kwargs):
        inplace=True
        for val_name, val in kwargs.items():
            if val_name == 'inplace':
                inplace=val
                kwargs.pop(val_name)
                break
        self.Convolution(name, **kwargs)
        if 1:
            self.BatchNorm(bn_name)
            self.Scale(None, inplace=inplace)
            self.ReLU(relu_name)
        else:
            self.BatchNorm(bn_name, inplace=inplace)            
            self.Scale(None)
            self.ReLU(relu_name)

    def conv_bn(self, name, bn_name=None, relu_name=None, **kwargs):
        inplace=True
        for val_name, val in kwargs.items():
            if val_name == 'inplace':
                inplace=val
                kwargs.pop(val_name)
                break
        self.Convolution(name, **kwargs)
        self.BatchNorm(bn_name)
        self.Scale(None, inplace=inplace)

    def softmax_acc(self,bottom, **kwargs):
        self.Softmax(bottom=[bottom])

        has_label=None
        for name, value in kwargs.items():
            if name == 'label':
                has_label = value
        if has_label is None:
            self.Accuracy()
        else:
            self.Accuracy(label=has_label)

    def Matmul(self):
        self.setup(self.suffix('AATI', None), 'Matmul', bottom=[self.this.name, self.this.name])
        self.include()
            

    #************************** network blocks **************************

    def res_func(self, name, num_output, up=False, orth=False, v2=False):
        if orth:
            if v2:
                orth_func = self.orth_loss_v2
            else:
                orth_func = self.orth_loss

            inplace = False
        else:
            inplace = True
        bottom = self.cur.name
        print(bottom)
        self.conv_bn_relu(name+'_conv0', num_output=num_output, stride=1+int(up), inplace=inplace)
        if orth:
            orth_func(name+'_conv0')
        self.conv_bn(name+'_conv1', num_output=num_output, inplace=inplace)
        if orth:
            orth_func(name+'_conv1')
        if up:
            self.conv_bn(name+'_proj', num_output=num_output, bottom=[bottom], pad=0, kernel_size=2, stride=2)
            if inplace:
                self.Eltwise(name+'_sum', bottom1=name+'_conv1')
            else:
                self.Eltwise(name+'_sum', bottom1=name+'_conv1_scale')
        else:
            if inplace:
                self.Eltwise(name+'_sum', bottom1=bottom)
            else:
                self.Eltwise(name+'_sum', bottom1=bottom, bottom0=name+'_conv1')
        if orth == False:
            pass
            self.ReLU()
    
    def res_group(self, group_id, n, num_output, orth=False, **kwargs):
        def name(block_id):
            return 'group{}'.format(group_id) + '_block{}'.format(block_id)

        if group_id == 0:
            up = False
        else:
            up = True
        self.res_func(name(0), num_output, up=up, orth=orth, **kwargs)
        for i in range(1, n):
            self.res_func(name(i), num_output, orth=orth, **kwargs)

    def plain_func(self, name, num_output, up=False, **kwargs):
        self.conv_bn_relu(name+'_conv0', num_output=num_output, stride=1+int(up), **kwargs)
        self.conv_bn_relu(name+'_conv1', num_output=num_output, **kwargs)
    
    # def orth_loss(self, bottom_name):
    #     # self.Python('orth_loss', 'orthLossLayer', loss_weight=1, bottom=[bottom_name], top=[name], name=name)
    #     # , bottom=[bottom+'_MVN']

    #     # save bottom
    #     mainpath = self.bottom

    #     bottom = bottom_name #'NormLayer', 
    #     # self.MVN(bottom=[bottom])
    #     layer = "TransposeLayer"
    #     layername = bottom_name+'_' + layer
    #     outputs = [layername]#, bottom_name+'_zerolike']
    #     self.Python(layer, layer, top=outputs, bottom=[bottom], name=layername, phase='TRAIN')
    #     self.Matmul()
    #     # layer="diagLayer"
    #     # layername = bottom_name+'_' + layer
        
    #     # self.Python(layer, layer, top=[layername], name=layername, phase='TRAIN')
    #     outputs = [self.this.name]#, bottom_name+'_zerolike']
    #     self.EuclideanLoss(name=bottom_name+'_euclidean', bottom=outputs, loss_weight=1e-3, phase='TRAIN')
        
    #     # restore bottom
    #     self.cur = mainpath

    def orth_loss_v2(self, bottom_name):
        # self.Python('orth_loss', 'orthLossLayer', loss_weight=1, bottom=[bottom_name], top=[name], name=name)
        # , bottom=[bottom+'_MVN']
        # save bottom
        mainpath = self.bottom

        bottom = bottom_name #'NormLayer', 
        # self.MVN(bottom=[bottom])
        layer = "TransposeLayer"
        layername = bottom_name+'_' + layer
        outputs = [layername]
        self.Python(layer, layer, top=outputs, bottom=[bottom], name=layername, phase='TRAIN')
        self.Matmul()
        
        outputs = [self.this.name]
        self.EuclideanLoss(name=bottom_name+'_euclidean', bottom=outputs, loss_weight=1e-1, phase='TRAIN')
        
        # restore bottom
        self.cur = mainpath



    def plain_orth_func(self, name, num_output, up=False, **kwargs):
        self.conv_bn_relu(name+'_conv0', num_output=num_output, stride=1+int(up), **kwargs)
        # if not up:
        #     # scale_param = 128 * 8**6 / num_output
        self.orth_loss_v2(name+'_conv0')
        self.conv_bn_relu(name+'_conv1', num_output=num_output, **kwargs)
        self.orth_loss_v2(name+'_conv1')
    
    def plain_group(self, group_id, n, num_output,orth=False, inplace=True, **kwargs):
        # if group_id == 0:
        #     num_output = 32
        def name(block_id):
            return 'group{}'.format(group_id) + '_block{}'.format(block_id)
        if orth:
            conv_func = self.plain_orth_func
        else:
            conv_func = self.plain_func
            bottom_name = None

        if group_id == 0:
            up = False
        else:
            up = True
        if n != 0:
            conv_func(name(0), num_output, up=up, inplace=inplace, **kwargs)
        else:
            self.conv_bn_relu(name(0)+'_conv0', num_output=num_output, stride=1+int(up), inplace=inplace, **kwargs)

        for i in range(1, n):
            conv_func(name(i), num_output, inplace=inplace, **kwargs)
    #************************** networks **************************
    def resnet_cifar(self, n=3, orth=False, num_output = 16, **kwargs):
        """6n+2, n=3 9 18 coresponds to 20 56 110 layers"""
        self.conv_bn_relu('first_conv', num_output=num_output)
        for i in range(3):
            self.res_group(i, n, num_output*(2**i), orth=orth, **kwargs)
        
        self.Pooling("global_avg_pool", global_pooling=True)
        self.InnerProduct()
        self.SoftmaxWithLoss()
        self.softmax_acc(bottom='fc')

    def plain_cifar(self, n=3,orth=False, inplace=True, num_output = 32, **kwargs):
        """6n+2, n=3 9 18 coresponds to 20 56 110 layers"""
        
        self.conv_bn_relu('first_conv', num_output=num_output, inplace=inplace)
        # self.orth_loss('first_conv')

        for i in range(3):
            self.plain_group(i, n, num_output*(2**i), orth=orth, inplace=inplace, **kwargs)
        
        self.Pooling("global_avg_pool", global_pooling=True)
        self.InnerProduct(**kwargs)
        self.SoftmaxWithLoss()
        self.softmax_acc(bottom='fc')

def resnet(n=3, num_output = 16):
    """6n+2, n=3 9 18 coresponds to 20 56 110 layers"""    
    net_name = "resnet-"    
    pt_folder = osp.join(osp.abspath(osp.curdir), net_name +str(6*n+2))
    name = net_name+str(6*n+2)+'-cifar10'

    if n > 18:
        # warm up
        solver = Solver(solver_name="solver_warm.prototxt", folder=pt_folder, lr_policy=Solver.policy.fixed)
        solver.p.base_lr = 0.01
        solver.set_max_iter(500)
        solver.write()
        del solver
    
    solver = Solver(folder=pt_folder)
    solver.write()
    del solver

    builder = Net(name)
    builder.Data('cifar-10-batches-py/train', phase='TRAIN', crop_size=32)
    builder.Data('cifar-10-batches-py/test', phase='TEST')
    builder.resnet_cifar(n, num_output=num_output)
    builder.write(folder=pt_folder)

def resnet_orth(n=3):
    """6n+2, n=3 9 18 coresponds to 20 56 110 layers"""    
    net_name = "resnet-orth-"    
    pt_folder = osp.join(osp.abspath(osp.curdir), net_name +str(6*n+2))
    name = net_name+str(6*n+2)+'-cifar10'

    if n > 18:
        # warm up
        solver = Solver(solver_name="solver_warm.prototxt", folder=pt_folder, lr_policy=Solver.policy.fixed)
        solver.p.base_lr = 0.01
        solver.set_max_iter(500)
        solver.write()
        del solver
    
    solver = Solver(folder=pt_folder)
    solver.write()
    del solver

    builder = Net(name)
    builder.Data('cifar-10-batches-py/train', phase='TRAIN', crop_size=32)
    builder.Data('cifar-10-batches-py/test', phase='TEST')
    builder.resnet_cifar(n, orth=True)
    builder.write(folder=pt_folder)

def resnet_orth_v2(n=3):
    """6n+2, n=3 9 18 coresponds to 20 56 110 layers"""    
    net_name = "resnet-orth-v2"    
    pt_folder = osp.join(osp.abspath(osp.curdir), net_name +str(6*n+2))
    name = net_name+str(6*n+2)+'-cifar10'

    if n > 18:
        # warm up
        solver = Solver(solver_name="solver_warm.prototxt", folder=pt_folder, lr_policy=Solver.policy.fixed)
        solver.p.base_lr = 0.01
        solver.set_max_iter(500)
        solver.write()
        del solver
    
    solver = Solver(folder=pt_folder)
    solver.write()
    del solver

    builder = Net(name)
    builder.Data('cifar-10-batches-py/train', phase='TRAIN', crop_size=32)
    builder.Data('cifar-10-batches-py/test', phase='TEST')
    builder.resnet_cifar(n, orth=True, v2=True)
    builder.write(folder=pt_folder)

def plain(n=3):
    """6n+2, n=3 9 18 coresponds to 20 56 110 layers"""
    net_name = "plain"
    pt_folder = osp.join(osp.abspath(osp.curdir), net_name +str(6*n+2))
    name = net_name+str(6*n+2)+'-cifar10'

    solver = Solver(folder=pt_folder)
    solver.write()
    del solver

    builder = Net(name)
    builder.Data('cifar-10-batches-py/train', phase='TRAIN', crop_size=32)
    builder.Data('cifar-10-batches-py/test', phase='TEST')
    builder.plain_cifar(n, num_output = 16)
    builder.write(folder=pt_folder)

def plain_orth(n=3):
    """6n+2, n=3 5 7 9 18 coresponds to 20 56 110 layers"""
    net_name = "plain-orth"
    pt_folder = osp.join(osp.abspath(osp.curdir), net_name +str(6*n+2))
    name = net_name+str(6*n+2)+'-cifar10'

    solver = Solver(folder=pt_folder)
    solver.write()
    del solver

    builder = Net(name)
    builder.Data('cifar-10-batches-py/train', phase='TRAIN', crop_size=32)
    builder.Data('cifar-10-batches-py/test', phase='TEST')
    builder.plain_cifar(n, orth=True)
    builder.write(folder=pt_folder)

def plain_orth_v1(n=3):
    """6n+2, n=3 5 7 9 18 coresponds to 20 32 44 56 110 layers"""
    net_name = "plain-orth-v1-"
    pt_folder = osp.join(osp.abspath(osp.curdir), net_name +str(6*n+2))
    name = net_name+str(6*n+2)+'-cifar10'

    solver = Solver(folder=pt_folder)
    solver.write()
    del solver

    builder = Net(name)
    builder.Data('cifar-10-batches-py/train', phase='TRAIN', crop_size=32)
    builder.Data('cifar-10-batches-py/test', phase='TEST')
    builder.plain_cifar(n, orth=True, inplace=False, num_output = 16)
    builder.write(folder=pt_folder)

def acc(n=3):
    """6n+2, n=3 9 18 coresponds to 20 56 110 layers"""
    net_name = "plain"
    pt_folder = osp.join(osp.abspath(osp.curdir), net_name +str(6*n+2))
    name = net_name+str(6*n+2)+'-cifar10'

    solver = Solver(folder=pt_folder)
    solver.write()
    del solver

    builder = Net(name)
    builder.Data('cifar-10-batches-py/train', phase='TRAIN', crop_size=32)
    builder.Data('cifar-10-batches-py/test', phase='TEST')
    builder.plain_cifar(n, num_output = 16, inplace=False)
    builder.write(folder=pt_folder)

if __name__ == '__main__':
    func = [resnet, plain, plain_orth, plain_orth_v1, resnet_orth, resnet_orth_v2]
    #        0        1         2           3           4
    func[1](17)
    # acc(0)

