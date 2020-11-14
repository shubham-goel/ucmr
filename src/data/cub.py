from __future__ import division, print_function

import os.path as osp

import numpy as np
import scipy.io as sio
import scipy.ndimage.interpolation
from absl import flags
from torch.utils.data import DataLoader, Dataset

from . import base as base_data

cub_dir = '/data3/shubham/birds_data/CUB_200_2011/'
if not osp.isdir(cub_dir):
    cub_dir = '/scratch/shubham/CUB_200_2011/'
if not osp.isdir(cub_dir):
    cub_dir = '/home/shubham/data/birds_data_rsync/birds_data/CUB_200_2011/'
if not osp.isdir(cub_dir):
    cub_dir = '/scratch3/shubham/data/birds_data/CUB_200_2011/'
if not osp.isdir(cub_dir):
    cub_dir = '/private/home/shubhamgoel/data/CUB_200_2011/'

flags.DEFINE_string('cub_dir', cub_dir, 'CUB Data Directory')
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '../..', 'cachedir')

flags.DEFINE_string('cub_cache_dir', osp.join(cache_path, 'cub'), 'CUB Data Directory')
flags.DEFINE_boolean('drop_last_batch', True, 'Whether to drop last (incomplete) batch')


class CubDataset(base_data.BaseDataset):

    def __init__(self, opts):
        super(CubDataset, self).__init__(opts,)
        self.data_dir = opts.cub_dir
        self.data_cache_dir = opts.cub_cache_dir
        self.opts = opts
        self.img_dir = osp.join(self.data_dir, 'images')
        self.anno_path = osp.join(self.data_cache_dir, 'data', '%s_cub_cleaned.mat' % opts.split)
        self.anno_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % opts.split)
        self.anno_train_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % 'train')
        self.jitter_frac = opts.jitter_frac
        self.padding_frac = opts.padding_frac
        self.img_size = opts.img_size
        if not osp.exists(self.anno_path):
            print('%s doesnt exist!' % self.anno_path)
            import ipdb
            ipdb.set_trace()

        # Load the annotation file.
        print('loading %s' % self.anno_path)
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.kp3d = sio.loadmat(self.anno_train_sfm_path, struct_as_record=False,
                                squeeze_me=True)['S'].transpose().copy()
        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1
        # self.kp_names = ['Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye',
        #                  'LLeg', 'LWing', 'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat']
        # self.mean_shape = sio.loadmat(osp.join(opts.cub_cache_dir, 'uv', 'mean_shape.mat'))
        # self.kp_uv = self.preprocess_to_find_kp_uv(self.kp3d, self.mean_shape['faces'], self.mean_shape[
        #                                            'verts'], self.mean_shape['sphere_verts'])
        self.flip = opts.flip
        return


def data_loader(opts):
    dset = CubDataset(opts)
    # dset = d_set_func(opts, filter_key=filter_key)
    return DataLoader(
        dset,
        batch_size=opts.batch_size,
        shuffle=opts.shuffle_data,
        num_workers=opts.n_data_workers,
        pin_memory=True,
        drop_last=opts.drop_last_batch,
        collate_fn=base_data.collate_fn)
