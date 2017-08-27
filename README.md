# Channel Pruning for Accelerating Very Deep Neural Networks
By [Yihui He](http://yihui-he.github.io/) (Xi'an Jiaotong University), Xiangyu Zhang and [Jian Sun](http://jiansun.org/) (Megvii)  
**ICCV 2017**  

In this repository, we illustrate channel pruning VGG-16 **4X** with our 3C method. After finetuning, the Top-5 accuracy is **89.9%**  (suffers no performance degradation).

![i2](http://yihui-he.github.io/assets_files/structure-1.png) | ![i1](http://yihui-he.github.io/assets_files/ill-1.png)
:-------------------------:|:-------------------------:
Structured simplification methods             |  Channel pruning (d)

### Citation
If you find the code useful in your research, please consider citing:

    @article{he2017channel,
      title={Channel Pruning for Accelerating Very Deep Neural Networks},
      author={He, Yihui and Zhang, Xiangyu and Sun, Jian},
      journal={arXiv preprint arXiv:1707.06168},
      year={2017}
    }
    
### Contents
1. [Requirements](#requirements)
2. [Installation](#installation-sufficient-for-the-demo)
3. [Channel Pruning and finetuning](#channel-pruning)  
4. [Pruned models for download](#pruned-models-for-download)

### requirements
1. Python3 packages you might not have: `scipy`, `sklearn`, `easydict`
2. For finetuning with 128 batch size, 4 GPUs (~11G of memory)

### Installation (sufficient for the demo)
1. Clone the repository
    ```Shell
    # Make sure to clone with --recursive
    git clone --recursive https://github.com/yihui-he/channel-pruning.git
    ```
2. Build my Caffe fork
    ```Shell
    cd caffe

    # If you're experienced with Caffe and have all of the requirements installed, then simply do:
    make -j8 && make pycaffe
    # Or follow the Caffe installation instructions here:
    # http://caffe.berkeleyvision.org/installation.html
    ```
3. Download ImageNet classification dataset
    http://www.image-net.org/download-images  
   Specify imagenet `source` path in `temp/vgg.prototxt` (line 12 and 36)
    
### Channel Pruning  
*For fast testing, you can directly download pruned model. See [next section](#pruned-models-for-download)*
1. Download the original VGG-16 model
    http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel  
   move it to `temp/vgg.caffemodel` (or create a softlink instead)

2. Start Channel Pruning
    ```Shell
    python3 train.py -action c3 -caffe GPU0
    # or log it with ./run.sh python3 train.py -action c3 -caffe GPU0
    ```
3. Combine some factorized layers for further compression, and calculate the acceleration ratio
    ```Shell
    ./combine.sh | xargs ./calflop.sh
    ```
    
4. Finetuning
    ```Shell
    ./finetune.sh GPU0,GPU1,GPU2,GPU3
    ```

5. Testing
    Though testing is done while finetuning, you can test anytime with:
    ```Shell
    caffe test -model path/to/prototxt -weights path/to/caffemodel -iterations 5000 -gpu GPU0
    ```
### Pruned models (for download)
  For fast testing, you can directly download pruned model from [release](https://github.com/yihui-he/channel-pruning/releases/tag/VGG-16_3C4x): https://github.com/yihui-he/channel-pruning/releases/download/VGG-16_3C4x/channel_pruning_VGG-16_3C4x.zip  
  Test with:
  
  ```Shell
  caffe test -model channel_pruning_VGG-16_3C4x.prototxt -weights channel_pruning_VGG-16_3C4x.caffemodel -iterations 5000 -gpu GPU0
  ```
