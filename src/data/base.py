from __future__ import absolute_import, division, print_function

import os.path as osp

import cv2
import numpy as np
import torch
import torchvision
from absl import app, flags
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from ..nnutils import geom_utils
from ..utils import image as image_utils
from ..utils import transformations

flags.DEFINE_integer('img_size', 256, 'image size')
flags.DEFINE_enum('split', 'train', ['train', 'val', 'all', 'test'], 'eval split')
flags.DEFINE_float('padding_frac', 0.05, 'bbox is increased by this fraction of max_dim')
flags.DEFINE_float('jitter_frac', 0.05, 'bbox is jittered by this fraction of max_dim')
flags.DEFINE_boolean('flip', True, 'Allow flip bird left right')
flags.DEFINE_boolean('tight_crop', False, 'Use Tight crops')
flags.DEFINE_boolean('flip_train', False, 'Mirror Images while training')
flags.DEFINE_integer('number_pairs', 10000,
                     'N random pairs from the test to check if the correspondence transfers.')
flags.DEFINE_integer('num_kps', 12, 'Number of keypoints')
flags.DEFINE_list('single_datapoint', '', 'Debug on single element (provide index)')
flags.DEFINE_boolean('use_cameraPoseDict_as_gt', False, 'Use precomputed camera poses in opts.cameraPoseDict as GT')
flags.DEFINE_boolean('cameraPoseDict_dataloader_isCamPose', True, 'Usual "campose" dict')
flags.DEFINE_string('cameraPoseDict_dataloader', '', 'Path to pre-computed camera pose dict for entire dataset, in the dataloader')
flags.DEFINE_list('cameraPoseDict_dataloader_mergewith', '', 'Merge camera poses from this dict, into original dict')
flags.DEFINE_boolean('dataloader_computeMaskDt', True, 'Comute mask_dt & mask_dtbarrier. Turn off to save load time')


def get_campose_dict(campose_path, is_campose):
    '''
    Returns campose dict: {frame_id: (cams_px7, scores_p, gt_st_3)}
    '''
    if is_campose:
        try:
            x = np.load(campose_path, allow_pickle=True)
            campose_dict = x['campose'].item()
        except UnicodeError:
            x = np.load(campose_path, allow_pickle=True, encoding='bytes')
            campose_dict = x['campose'].item()
    else:
        x = np.load(campose_path, allow_pickle=True)
        campose_dict = {}
        gt_cam_nx7 = torch.as_tensor(x['gt_cam'])
        cams_nxpx7 = torch.as_tensor(x['cam_values'])
        score_nxp = torch.as_tensor(x['quat_scores'])
        fids_nx1 = torch.as_tensor(x['frame_id'])

        gt_cam_flip_nx7 = geom_utils.reflect_cam_pose(gt_cam_nx7)
        cams_flip_nxpx7 = geom_utils.reflect_cam_pose(cams_nxpx7)
        fids_flip_nx1 = int(1e6) - fids_nx1

        flip = fids_nx1>int(1e6)/2
        gt_cam_nx7 = torch.where(flip, gt_cam_flip_nx7, gt_cam_nx7)
        cams_nxpx7 = torch.where(flip[:,:,None], cams_flip_nxpx7, cams_nxpx7)
        fids_nx1 = torch.where(flip, fids_flip_nx1, fids_nx1)

        assert((fids_nx1>=0).all())
        assert((fids_nx1<int(1e6)/2).all())

        for i in range(fids_nx1.shape[0]):
            fid = int(fids_nx1[i,0])
            gt_st_3 = gt_cam_nx7[i,0:3]
            cams_px7 = cams_nxpx7[i,:]
            score_p = score_nxp[i,:]
            campose_dict[fid] = (cams_px7, score_p, gt_st_3)

    return campose_dict

class BaseDataset(Dataset):

    def __init__(self, opts):
        # Child class should define/load:
        # self.kp_perm
        # self.img_dir
        # self.anno
        # self.anno_sfm
        self.opts = opts
        self.flip = opts.flip
        self.img_size = opts.img_size
        self.jitter_frac = opts.jitter_frac
        self.padding_frac = opts.padding_frac
        self.rngFlip = np.random.RandomState(0)
        self.flip_transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                        torchvision.transforms.RandomHorizontalFlip(1),
                                        torchvision.transforms.ToTensor()])

        if opts.use_cameraPoseDict_as_gt:
            self.cameraPoseDict = get_campose_dict(opts.cameraPoseDict_dataloader, opts.cameraPoseDict_dataloader_isCamPose)
            print(f'Loaded cam_pose_dict of size {len(self.cameraPoseDict)} (should be 5964) in dataloader')
            if opts.cameraPoseDict_dataloader_mergewith:
                for _ll in opts.cameraPoseDict_dataloader_mergewith:
                    cameraPoseDict2 = get_campose_dict(_ll, opts.cameraPoseDict_dataloader_isCamPose)
                    self.cameraPoseDict = {**cameraPoseDict2, **self.cameraPoseDict}
                    print(f'Merged cam_pose_dict of size {len(self.cameraPoseDict)} (should be 5964) in dataloader')

        return

    def get_anno(self, index):
        data = self.anno[index]
        data_sfm = self.anno_sfm[index]
                # sfm_pose = (sfm_c, sfm_t, sfm_r)
        sfm_pose = [np.copy(data_sfm.scale), np.copy(data_sfm.trans), np.copy(data_sfm.rot)]

        sfm_rot = np.pad(sfm_pose[2], (0, 1), 'constant')
        sfm_rot[3, 3] = 1
        sfm_pose[2] = transformations.quaternion_from_matrix(sfm_rot, isprecise=True)

        img_path = osp.join(self.img_dir, str(data.rel_path))

        # Adjust to 0 indexing
        bbox = np.array(
            [data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2],
            float) - 1

        parts = data.parts.T.astype(float)
        kp = np.copy(parts)
        vis = kp[:, 2] > 0
        kp[vis, :2] -= 1

        return img_path, data.mask, bbox, sfm_pose, kp, vis

    def forward_img(self, index):

        img_path, mask, bbox, sfm_pose, kp, vis = self.get_anno(index)

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img = img / 255.0

        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        assert(img.shape[:2] == mask.shape)
        mask = np.expand_dims(mask, 2)

        # Peturb bbox
        if self.opts.tight_crop:
            self.padding_frac = 0.0

        if self.opts.split == 'train':
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=self.jitter_frac)
        else:
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=0)
        if self.opts.tight_crop:
            bbox = bbox
        else:
            bbox = image_utils.square_bbox(bbox)

        # crop image around bbox, translate kps
        img, mask, kp, sfm_pose = self.crop_image(img, mask, bbox, kp, vis, sfm_pose)


        # scale image, and mask. And scale kps.
        if self.opts.tight_crop:
            img, mask, kp, sfm_pose = self.scale_image_tight(img, mask, kp, vis, sfm_pose)
        else:
            img, mask, kp, sfm_pose = self.scale_image(img, mask, kp, vis, sfm_pose)


        # Mirror image on random.
        if self.opts.split == 'train':
            flipped, img, mask, kp, sfm_pose = self.mirror_image(img, mask, kp, sfm_pose)
        else:
            flipped = False

        # Normalize kp to be [-1, 1]
        img_h, img_w = img.shape[:2]
        kp_norm, sfm_pose = self.normalize_kp(kp, sfm_pose, img_h, img_w)

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))

        return flipped, img, kp_norm, mask, sfm_pose

    def normalize_kp(self, kp, sfm_pose, img_h, img_w):
        if kp is not None:
            vis = kp[:, 2, None] > 0
            kp = np.stack([2 * (kp[:, 0] / img_w) - 1,
                            2 * (kp[:, 1] / img_h) - 1,
                            kp[:, 2]]).T
            kp = vis * kp
        sfm_pose[0] *= (1.0 / img_w + 1.0 / img_h)
        sfm_pose[1][0] = 2.0 * (sfm_pose[1][0] / img_w) - 1
        sfm_pose[1][1] = 2.0 * (sfm_pose[1][1] / img_h) - 1

        return kp, sfm_pose

    def crop_image(self, img, mask, bbox, kp, vis, sfm_pose):
        # crop image and mask and translate kps
        img = image_utils.crop(img, bbox, bgval=1)
        mask = image_utils.crop(mask, bbox, bgval=0)
        if kp is not None:
            assert(vis is not None)
            kp[vis, 0] -= bbox[0]
            kp[vis, 1] -= bbox[1]

            kp[vis,0] = np.clip(kp[vis,0], a_min=0, a_max=bbox[2] -bbox[0])
            kp[vis,1] = np.clip(kp[vis,1], a_min=0, a_max=bbox[3] -bbox[1])

        sfm_pose[1][0] -= bbox[0]
        sfm_pose[1][1] -= bbox[1]

        return img, mask, kp, sfm_pose

    def scale_image_tight(self, img, mask, kp, vis, sfm_pose):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[1]
        bheight = np.shape(img)[0]

        scale_x = self.img_size/bwidth
        scale_y = self.img_size/bheight

        # scale = self.img_size / float(max(bwidth, bheight))
        # pdb.set_trace()
        img_scale = cv2.resize(img, (self.img_size, self.img_size))
        # img_scale, _ = image_utils.resize_img(img, scale)
        # if img_scale.shape[0] != self.img_size:
        #     print('bad!')
        #     import ipdb; ipdb.set_trace()
        # mask_scale, _ = image_utils.resize_img(mask, scale)

        mask_scale = cv2.resize(mask, (self.img_size, self.img_size))

        if kp is not None:
            assert(vis is not None)
            kp[vis, 0:1] *= scale_x
            kp[vis, 1:2] *= scale_y
        sfm_pose[0] *= scale_x
        sfm_pose[1] *= scale_y

        return img_scale, mask_scale, kp, sfm_pose

    def scale_image(self, img, mask, kp, vis, sfm_pose):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = self.img_size / float(max(bwidth, bheight))
        img_scale, _ = image_utils.resize_img(img, scale)
        # if img_scale.shape[0] != self.img_size:
        #     print('bad!')
        #     import ipdb; ipdb.set_trace()
        mask_scale, _ = image_utils.resize_img(mask, scale)
        if kp is not None:
            assert(vis is not None)
            kp[vis, :2] *= scale
        sfm_pose[0] *= scale
        sfm_pose[1] *= scale

        return img_scale, mask_scale, kp, sfm_pose

    def mirror_image(self, img, mask, kp, sfm_pose):
        if self.rngFlip.rand(1) > 0.5 and self.flip:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()

            if kp is not None:
                # Flip kps.
                new_x = img.shape[1] - kp[:, 0] - 1
                kp = np.hstack((new_x[:, None], kp[:, 1:]))
                kp = kp[self.kp_perm, :]
                # kp_uv_flip = kp_uv[self.kp_perm, :]
            # Flip sfm_pose Rot.
            R = transformations.quaternion_matrix(sfm_pose[2])
            flip_R = np.diag([-1, 1, 1, 1]).dot(R.dot(np.diag([-1, 1, 1, 1])))
            sfm_pose[2] = transformations.quaternion_from_matrix(flip_R, isprecise=True)
            # Flip tx
            tx = img.shape[1] - sfm_pose[1][0] - 1
            sfm_pose[1][0] = tx
            return True, img_flip, mask_flip, kp, sfm_pose
        else:
            return False, img, mask, kp, sfm_pose

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        # if index == 1452:
        #     pdb.set_trace()
        datapoints = self.opts.single_datapoint
        datapoints = [int(x) for x in datapoints]
        if len(datapoints) > 0:
            index = datapoints[index % len(datapoints)]

        flipped, img, kp, mask, sfm_pose = self.forward_img(index)

        if flipped:
            index = int(1e6) - index

        # anchor, positive_samples, negative_samples = self.forward_random_pixel_samples(img, mask, count=10, margin=10)
        sfm_pose[0].shape = 1
        elem = {
            'img': img,
            'kp': 0 if kp is None else kp,
            # 'kp_uv': kp_uv,
            'mask': mask,
            'sfm_pose': np.concatenate(sfm_pose),  # scale (1), trans (2), quat(4)
            'inds': np.array([index]),
            # 'anchor': anchor,
            # 'pos_inds': positive_samples,
            # 'neg_inds': negative_samples,
        }
        if self.opts.dataloader_computeMaskDt:
            elem['mask_dt'] = image_utils.compute_dt(mask)
            elem['mask_dt_barrier'] = image_utils.compute_dt_barrier(mask)

        if self.opts.flip_train:
            # flip_img = self.flip_transform((img.transpose(1,2,0)*255).astype(np.uint8))
            flip_img  = img[:, :, ::-1].copy()
            elem['flip_img'] = flip_img
            # elem['flip_img'] = img[:,:,-1::-1].copy()
            # flip_mask = self.flip_transform((mask[None, :, :].transpose(1,2,0)*225).astype(np.uint8))
            flip_mask = mask[:, ::-1].copy()
            elem['flip_mask'] = flip_mask
            if self.opts.dataloader_computeMaskDt:
                elem['flip_mask_dt'] = image_utils.compute_dt(flip_mask)
                elem['flip_mask_dt_barrier'] = image_utils.compute_dt_barrier(flip_mask)

        if self.opts.use_cameraPoseDict_as_gt:
            if flipped:
                cams, scores, gt_st1 = self.cameraPoseDict[int(1e6) - index]
                gt_st0 = elem['sfm_pose'][:3] * np.array([1,-1,1], dtype=np.float32)
            else:
                cams, scores, gt_st1 = self.cameraPoseDict[index]
                gt_st0 = elem['sfm_pose'][:3]
            cams = cams.numpy()
            scores = scores.numpy()
            gt_st1 = gt_st1.numpy()
            cam_id = np.argmax(scores, axis=0)
            best_cam = cams[cam_id]
            scale_factor = gt_st0[0] / gt_st1[0]
            scale = best_cam[0:1] * scale_factor
            trans = (best_cam[1:3] - gt_st1[1:3]) * scale_factor + gt_st0[1:3]
            best_cam = np.concatenate((scale, trans, best_cam[3:]), axis=0)
            if flipped:
                best_cam = best_cam * np.array([1, -1, 1, 1, 1, -1, -1], dtype=np.float32)
            elem['sfm_pose'] = best_cam

        return elem





def collate_fn(batch):
    '''Globe data collater.

    Assumes each instance is a dict.
    Applies different collation rules for each field.

    Args:
        batch: List of loaded elements via Dataset.__getitem__
    '''
    collated_batch = {'empty': True}
    # iterate over keys
    # new_batch = []
    # for valid,t in batch:
    #     if valid:
    #         new_batch.append(t)
    #     else:
    #         'Print, found a empty in the batch'

    # # batch = [t for t in batch if t is not None]
    # # pdb.set_trace()
    # batch = new_batch
    if len(batch) > 0:
        for key in batch[0]:
            collated_batch[key] = default_collate([elem[key] for elem in batch])
        collated_batch['empty'] = False
    return collated_batch
