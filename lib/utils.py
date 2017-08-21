from __future__ import print_function
import os.path as osp
import time
from termcolor import colored
import numpy as np
from IPython import embed
import lib.cfgs as cfgs
import os
import subprocess

cnt = 0

def printstage(*sentence):
    global cnt
    mystring =''
    for i in sentence:
        if not isinstance(i, str):
            i = str(i)
        mystring += i
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("stage"+str(cnt)+" "+mystring)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    cnt+=1

def mem_pt(pt):
    folder_file = osp.split(pt)
    mem = osp.join(folder_file[0], '_'.join(['mem',folder_file[1]]))
    return mem

def bn_pt(pt):
    folder_file = osp.split(pt)
    bn = osp.join(folder_file[0], '_'.join(['bn',folder_file[1]]))
    return bn

def x_pt(pt, x):
    folder_file = osp.split(pt)
    bn = osp.join(folder_file[0], '_'.join([x,folder_file[1]]))
    return bn

def beep():
    print('\a')

def arr2strarr(*kwargs):
    mylist = []
    for i in kwargs:
        if isinstance(i, str):
            mylist.append(i)
            continue
        mylist.append(str(i))
    return mylist

def underline(*kwargs):
    strarr = arr2strarr(*kwargs)
    return '_'.join(strarr)

def space(*kwargs):
    strarr = arr2strarr(*kwargs)
    return ' '.join(strarr)

def green(sentence):
    return colored(sentence, 'green')

def red(sentence):
    return colored(sentence, 'red')

def redprint(sentence):
    print(red(space(sentence)))

def OK(sentence):
    print(sentence + green(" [ OK ]"))

def FAIL(sentence):
    print(sentence + red(" [ FAIL ]"))

def CHECK_EQ(fake, real):

    for i, j in zip(fake.flatten(), real.flatten()):
        if np.abs(i-j) > 1e-4:
            FAIL(space(i, j))
            return False
    OK("CHECK_EQ")
    return True

def set_tf():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfgs.tf_vis

def set_caffe():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfgs.caffe_vis

def shell(*s):
    s = space(*s)
    print("$", s)
    print(os.system(s))
    #print(subprocess.check_output(s, shell=True))
    return 
    # return os.system(s)


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self,show=None, average=False):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if show is not None:
            print(show,self.diff)
        if average:
            return self.average_time
        else:
            return self.diff
    
if __name__ == "__main__":
    print(underline("a", "b"))
