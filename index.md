# [Channel Pruning for Accelerating Very Deep Neural Networks](https://arxiv.org/abs/1707.06168)
**ICCV 2017**, by [Yihui He](http://yihui-he.github.io/), [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en&oi=ao) and [Jian Sun](http://jiansun.org/)

Please have a look our new works on compressing deep models:
- [AMC: AutoML for Model Compression and Acceleration on Mobile Devices](http://openaccess.thecvf.com/content_ECCV_2018/html/Yihui_He_AMC_Automated_Model_ECCV_2018_paper.html) **ECCV'18**, which combines channel pruning and reinforcement learning to further accelerate CNN. [code](https://github.com/mit-han-lab/amc-release) and [models](https://github.com/mit-han-lab/amc-compressed-models) are available!
- [AddressNet: Shift-Based Primitives for Efficient Convolutional Neural Networks](https://arxiv.org/abs/1809.08458) **WACV'19**. We propose a family of efficient networks based on Shift operation. 
- [MoBiNet: A Mobile Binary Network for Image Classification](https://arxiv.org/abs/1907.12629) **WACV'20** Binarized MobileNets.

In this repository, we released code for the following models:

model | Speed-up | Accuracy
:-------------------------:|:-------------------------:|:-------------------------
[VGG-16 channel pruning](https://github.com/yihui-he/channel-pruning/releases/tag/channel_pruning_5x) |5x            |  88.1 (Top-5), 67.8 (Top-1)
[VGG-16 3C](https://github.com/yihui-he/channel-pruning/releases/tag/VGG-16_3C4x)<sup>1</sup>   |4x            |  89.9 (Top-5), 70.6 (Top-1)
[ResNet-50](https://github.com/yihui-he/channel-pruning/releases/tag/ResNet-50-2X) |2x |90.8 (Top-5), 72.3 (Top-1)
[faster RCNN](https://github.com/yihui-he/channel-pruning/releases/tag/faster-RCNN-2X4X)|  2x | 36.7 (AP@.50:.05:.95)
[faster RCNN](https://github.com/yihui-he/channel-pruning/releases/tag/faster-RCNN-2X4X)|  4x | 35.1 (AP@.50:.05:.95)

###### <sup>1</sup> 3C method combined spatial decomposition ([Speeding up Convolutional Neural Networks with Low Rank Expansions](https://arxiv.org/abs/1405.3866)) and channel decomposition ([Accelerating Very Deep Convolutional Networks for Classification and Detection](https://arxiv.org/abs/1505.06798)) (mentioned in 4.1.2) 



![i2](http://yihui-he.github.io/assets_files/structure-1.png) | ![i1](http://yihui-he.github.io/assets_files/ill-1.png)
:-------------------------:|:-------------------------:
Structured simplification methods             |  Channel pruning (d)

### Citation
If you find the code useful in your research, please consider citing:

    @InProceedings{He_2017_ICCV,
    author = {He, Yihui and Zhang, Xiangyu and Sun, Jian},
    title = {Channel Pruning for Accelerating Very Deep Neural Networks},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
    }
    
### Contents
1. [Requirements](#requirements)
2. [Installation](#installation-sufficient-for-the-demo)
3. [Channel Pruning and finetuning](#channel-pruning)  
4. [Pruned models for download](#pruned-models-for-download)
5. [Pruning faster RCNN](#pruning-faster-rcnn)
6. [FAQ](#faq)

### requirements
1. Python3 packages you might not have: `scipy`, `sklearn`, `easydict`, use `sudo pip3 install` to install.
2. For finetuning with 128 batch size, 4 GPUs (~11G of memory)

### Installation (sufficient for the demo)
1. Clone the repository
    ```Shell
    # Make sure to clone with --recursive
    git clone --recursive https://github.com/yihui-he/channel-pruning.git
    ```
2. Build [my Caffe](https://github.com/yihui-he/caffe-pro) fork (which support bicubic interpolation and resizing image shorter side to 256 then crop to 224x224) 
    ```Shell
    cd caffe

    # If you're experienced with Caffe and have all of the requirements installed, then simply do:
    make all -j8 && make pycaffe
    # Or follow the Caffe installation instructions here:
    # http://caffe.berkeleyvision.org/installation.html

    # you might need to add pycaffe to PYTHONPATH, if you've already had a caffe before
    ```
    
3. Download ImageNet classification dataset
    http://www.image-net.org/download-images  
    
4. Specify imagenet `source` path in `temp/vgg.prototxt` (line 12 and 36)
    
### Channel Pruning  
*For fast testing, you can directly download pruned model. See [next section](#pruned-models-for-download)*
1. Download the original VGG-16 model
    http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel  
   move it to `temp/vgg.caffemodel` (or create a softlink instead)

2. Start Channel Pruning
    ```Shell
    python3 train.py -action c3 -caffe [GPU0]
    # or log it with ./run.sh python3 train.py -action c3 -caffe [GPU0]
    # replace [GPU0] with actual GPU device like 0,1 or 2
    ```
3. Combine some factorized layers for further compression, and calculate the acceleration ratio.
   Replace the ImageData layer of `temp/cb_3c_3C4x_mem_bn_vgg.prototxt` with [`temp/vgg.prototxt`'s](https://github.com/yihui-he/channel-pruning/blob/master/temp/vgg.prototxt#L1-L49)
    ```Shell
    ./combine.sh | xargs ./calflop.sh
    ```
    
4. Finetuning
    ```Shell
    caffe train -solver temp/solver.prototxt -weights temp/cb_3c_vgg.caffemodel -gpu [GPU0,GPU1,GPU2,GPU3]
    # replace [GPU0,GPU1,GPU2,GPU3] with actual GPU device like 0,1,2,3
    ```

5. Testing

    Though testing is done while finetuning, you can test anytime with:
    ```Shell
    caffe test -model path/to/prototxt -weights path/to/caffemodel -iterations 5000 -gpu [GPU0]
    # replace [GPU0] with actual GPU device like 0,1 or 2
    ```
### Pruned models (for download)
  For fast testing, you can directly download pruned model from [release](https://github.com/yihui-he/channel-pruning/releases): 
  [VGG-16 3C 4X](https://github.com/yihui-he/channel-pruning/releases/tag/VGG-16_3C4x), [VGG-16 5X](https://github.com/yihui-he/channel-pruning/releases/tag/channel_pruning_5x), [ResNet-50 2X](https://github.com/yihui-he/channel-pruning/releases/tag/ResNet-50-2X). Or follow Baidu Yun [Download link](https://pan.baidu.com/s/1c2evwTa)
  
  Test with:
  
  ```Shell
  caffe test -model channel_pruning_VGG-16_3C4x.prototxt -weights channel_pruning_VGG-16_3C4x.caffemodel -iterations 5000 -gpu [GPU0]
  # replace [GPU0] with actual GPU device like 0,1 or 2
  ```
### Pruning faster RCNN
For fast testing, you can directly download pruned model from [release](https://github.com/yihui-he/channel-pruning/releases/tag/faster-RCNN-2X4X)  
Or you can:
1. clone my py-faster-rcnn repo: https://github.com/yihui-he/py-faster-rcnn
2. use the [pruned models](https://github.com/yihui-he/channel-pruning/releases/tag/faster-RCNN-2X4X) from this repo to train faster RCNN 2X, 4X, solver prototxts are in https://github.com/yihui-he/py-faster-rcnn/tree/master/models/pascal_voc

### FAQ
You can find answers of some commonly asked questions in our [Github wiki](https://github.com/yihui-he/channel-pruning/wiki), or just create a [new issue](https://github.com/yihui-he/channel-pruning/issues/new)