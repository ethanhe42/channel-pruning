from easydict import EasyDict as edict
c=edict()
gpu=1
dataset="imagenet" # cifar10
caffe_vis = '0,1,2,3'
tf_vis = '4,5,6,7'
accname = None
frozenname= None
layer = False # this might be for single layer evalation? Is called in several methods, including c3.solve() -by Mario
gt_feats = False # gt stands for ground truth -by Mario
_points_dict_name = "points_dict"
noTF = False
noTheano = True
imagenet_val = "path/to/caffe/examples/imagenet/ilsvrc12_val_lmdb"
cifar10_val = 'path/to/cifar-10-batches-py/test'
caffe_path = "path/to/caffe/build/tools/caffe"
mp=0
alpha=1e-3 #2e-5 # 1e-2
# resnet56 14% 7e-4
class Action:
    train='train'
    layer='layer'
    cifar='cifar'
    addbn='addbn'
    splitrelu='splitrelu'
    c3='c3'
    combine='combine'
class datasets:
    cifar='cifar10'
    imagenet='imagenet'
class kernels:
    dic='dic'
    pruning='pruning'
class resnet:
    solve_sum=True
    solve_conv=True
class Feats:
    decompose_method = 1
    inplace = False
class solvers:
    lightning='lightning'
    sk = 'sklearn'
    lowparams = 'lowparams'
    gd = 'gd'
    keras = 'keras'
    tls = "tls"
class pruning_options: # TODO: Consider adding another pruning option for alexnet, i.e. alexnet=2 -by Mario
    prb=0
    vgg=3
    resnet=4
    single=10
class Data:
    lmdb='lmdb'
    pro='pro'

class Models: # TODO: Consider adding a new attribute to the this class for alexnet -by Mario
    vgg='vgg'
    xception='xception'
    resnet='resnet'
    rescifar='rescifar'

class vgg:  # TODO: Consider adding a new class to hold information about alexnet -by Mario
    model='temp/vgg.prototxt'
    weights='temp/vgg.caffemodel'
    accname='accuracy@5'
    flop=15346630656 # FLOP of a model can be calculated with calflop.sh -by Mario

c.dic = edict()
c.dic.option=pruning_options.prb
c.dic.layeralpha=1
c.dic.debug=0
c.dic.afterconv=False
c.dic.fitfc=0
c.dic.keep = 3.
c.dic.rank_tol = .1
c.dic.prepooling = 1
c.dic.alter=0
c.dic.vh=1  # The excution of purning depends on this flag. However it also determines the execution of VH decomposition -by Mario
           # How to unlink the execution of both algoritms? -by Mario
# single layer
c.an = edict()
c.an.l1 = '' #'conv1_1'
c.an.l2 = '' #'conv1_2'
c.an.ratio = 2
c.an.filter = 0

# resnet
c.res=edict()
c.res.short = 0
c.res.bn = 1

# vh
#c.vh=edict()
#c.vh.ls=0

c.Action = Action.train
c.mp=True
c.kernelname='dic'
c.fc_ridge = 0
c.ls='linear'
c.nonlinear_fc = 0
c.nofc=0
c.splitconvrelu=True
c.nBatches=500
c.ntest = 0
c.nBatches_fc=c.nBatches * 10
c.frozen=0
c.nPointsPerLayer=10
c.fc_reg=True
c.autodet = False
c.solver=solvers.sk
c.shm='/tmp' #'/dev/shm'
c.log='logs/'
c.data=Data.lmdb
c.model=''
c.weights= 'temp/vgg.caffemodel'
c.prototxt= 'temp/vgg.prototxt'

def set_nBatches(n):
    c.nBatches = n
    c.nBatches_fc=c.nBatches# * 10
