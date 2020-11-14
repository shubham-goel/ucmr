"""
Data loader for pascal VOC categories.
Should output:
    - img: B X 3 X H X W
    - kp: B X nKp X 2
    - mask: B X H X W
    - sfm_pose: B X 7 (s, tr, q)
    (kp, sfm_pose) correspond to image coordinates in [-1, 1]
"""

from __future__ import absolute_import, division, print_function

import itertools
import os.path as osp

import numpy as np
import scipy.io as sio
from absl import app, flags
from torch.utils.data import DataLoader

from . import base as base_data

# -------------- flags ------------- #
# ---------------------------------- #
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '../../', 'cachedir')

cub_dir = '/data3/shubham/Imagenet/'
if not osp.isdir(cub_dir):
    cub_dir = '/home/shubham/data/Imagenet/'
if not osp.isdir(cub_dir):
    cub_dir = '/scratch3/shubham/data/Imagenet/'

imnet_class2sysnet = {'horse' : 'n02381460', 'zebra': 'n02391049' , 'bear':'n02131653', 'sheep': 'n10588074', 'cow': 'n01887787'}
flags.DEFINE_string('imnet_dir', cub_dir, 'Imagenet Data Directory')
flags.DEFINE_string('imnet_anno_path', osp.join(cache_path, 'imnet'), 'Directory where pascal annotations are saved')
flags.DEFINE_string('imnet_class', 'horse', 'Imagenet category name')
flags.DEFINE_string('imnet_cache_dir', osp.join(cache_path, 'imnet'), 'P3D Data Directory')

opts = flags.FLAGS

# -------------- Dataset ------------- #
# ------------------------------------ #


class ImnetDataset(base_data.BaseDataset):
    '''
    Imnet Data loader
    '''


    def __init__(self, opts,):
        super(ImnetDataset, self).__init__(opts,)
        sysnetId = imnet_class2sysnet[opts.imnet_class]
        self.img_dir = osp.join(opts.imnet_dir, 'ImageSets',sysnetId)
        self.data_cache_dir = opts.imnet_cache_dir
        imnet_cache_dir =  osp.join(opts.imnet_cache_dir, sysnetId)


        self.anno_path = osp.join(self.data_cache_dir, 'data', '{}_{}.mat'.format(sysnetId, opts.split))
        self.anno_sfm_path = osp.join(self.data_cache_dir, 'sfm', '{}_{}.mat'.format(sysnetId, opts.split))

        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.kp_perm = np.linspace(0, 9, 10).astype(np.int)
        self.kp_names = ['lpsum' for _ in range(len(self.kp_perm))]
        self.kp_uv = np.random.uniform(0,1, (len(self.kp_perm), 2))
        opts.num_kps = len(self.kp_perm)
        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        self.flip = opts.flip
        return

def imnet_dataloader(opts):
    dset = ImnetDataset(opts)
    return DataLoader(
        dset,
        batch_size=opts.batch_size,
        shuffle=opts.shuffle_data,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)

