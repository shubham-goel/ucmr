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

if osp.exists('/data1/shubhtuls'):
    kData = '/data1/shubhtuls/cachedir/PASCAL3D+_release1.1'
elif osp.exists('/scratch1/storage'):
    kData = '/scratch1/storage/PASCAL3D+_release1.1'
elif osp.exists('/home/shubham/data/'):
    kData = '/home/shubham/data/PASCAL3D+_release1.1'
else:  # Savio
    kData = '/global/home/users/kanazawa/scratch/PASCAL3D+_release1.1'

flags.DEFINE_string('p3d_dir', kData, 'PASCAL Data Directory')
flags.DEFINE_string('p3d_anno_path', osp.join(cache_path, 'p3d'), 'Directory where pascal annotations are saved')
flags.DEFINE_string('p3d_class', 'aeroplane', 'PASCAL VOC category name')

opts = flags.FLAGS

# -------------- Dataset ------------- #
# ------------------------------------ #
class P3dDataset(base_data.BaseDataset):
    '''
    VOC Data loader
    '''

    def __init__(self, opts, filter_key=None):
        super().__init__(opts)

        self.flip = opts.flip

        self.img_dir = osp.join(opts.p3d_dir, 'Images')
        self.kp_path = osp.join(
            opts.p3d_anno_path, 'data', '{}_kps.mat'.format(opts.p3d_class))
        self.anno_path = osp.join(
            opts.p3d_anno_path, 'data', '{}_{}.mat'.format(opts.p3d_class, opts.split))
        self.anno_sfm_path = osp.join(
            opts.p3d_anno_path, 'sfm', '{}_{}.mat'.format(opts.p3d_class, opts.split))

        # Load the annotation file.
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']
        self.kp_perm = sio.loadmat(
            self.kp_path, struct_as_record=False, squeeze_me=True)['kp_perm_inds'] - 1

        opts.num_kps = len(self.kp_perm)
        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)


#----------- Data Loader ----------#
#----------------------------------#

def data_loader(opts, shuffle=True):
    dset = P3dDataset(opts)
    # dset = d_set_func(opts, filter_key=filter_key)
    return DataLoader(
        dset,
        batch_size=opts.batch_size,
        shuffle=opts.shuffle_data,
        num_workers=opts.n_data_workers,
        pin_memory=True,
        drop_last=opts.drop_last_batch,
        collate_fn=base_data.collate_fn)
