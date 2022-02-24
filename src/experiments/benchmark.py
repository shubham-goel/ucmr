"""Testing File
python -m src.experiments.benchmark \
    --pred_pose \
    --pretrained_network_path=cachedir/snapshots/cam/e400_cub_train_cam4/pred_net_600.pth \
    --shape_path=cachedir/template_shape/bird_template.npy \
    --nodataloader_computeMaskDt \
    --split=test

The --render option renders results from the entire dataset

"""

from __future__ import absolute_import, division, print_function

import os

import matplotlib as mpl
from absl import app, flags

if 'DISPLAY' not in os.environ:
    print('Display not found. Using Agg backend for matplotlib')
    mpl.use('Agg')
else:
    print(f'Found display : {os.environ["DISPLAY"]}')
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scipy_stats
import torch
import torchvision
from absl import flags
from imageio import imwrite
from pathlib2 import Path
from tqdm import tqdm
import scipy.io as sio

from ..data import cub as cub_data
from ..data import imagenet as imagenet_data
from ..data import json_dataset as json_data
from ..data import p3d as p3d_data
from ..nnutils import geom_utils
from ..nnutils import predictor as pred_utils
from ..utils import bird_vis
from ..utils import image as image_utils
from ..utils import mesh
from ..utils import transformations, visutil

#-------------- flags -------------#
#----------------------------------#
## Flags for training
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '../../', 'cachedir')

flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_string('results_dir', 'results/', 'Results directory')
flags.DEFINE_string('dataset', 'cub', 'yt (YouTube), or cub, or yt_filt (Youtube, refined)')
flags.DEFINE_string('pretrained_network_path', '', 'If empty, will use "cache_dir/name".')
flags.DEFINE_float('initial_quat_bias_deg', 90, 'Rotation bias in deg. 90 for head-view, 45 for breast-view')

flags.DEFINE_integer('batch_size', 128, 'Size of minibatches')
flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')
flags.DEFINE_boolean('shuffle_data', False, 'Whether dataloader shuffles data')
flags.DEFINE_integer('max_eval_iter', 0, 'Maximum evaluation iterations. 0 => 1 epoch.')
flags.DEFINE_boolean('render', False, 'Render data')
flags.DEFINE_boolean('save_mats', False, 'Save mat files for 3D iou evaluation')

class Tester(object):
    def __init__(self, opts):
        self.opts = opts

        self.results_dir = f'{opts.results_dir}/{opts.dataset}'
        self.render_dir = f'{opts.results_dir}/{opts.dataset}/render/'
        Path(self.render_dir).mkdir(parents=True, exist_ok=True)

        self.predictor = pred_utils.MeshPredictor(opts)
        self.faces = self.predictor.faces

        # Initialize dataset
        self.init_dataset()

    def init_dataset(self):
        opts = self.opts
        if opts.dataset == 'cub':
            dataloader_fn = cub_data.data_loader
        elif opts.dataset == 'imnet':
            dataloader_fn =  imagenet_data.imnet_dataloader
        elif opts.dataset == 'p3d':
            dataloader_fn =  p3d_data.data_loader
        elif opts.dataset == 'json':
            dataloader_fn =  json_data.data_loader
        else:
            raise ValueError('Unknown dataset %d!' % opts.dataset)

        self.dataloader = dataloader_fn(opts)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=image_utils.BGR_MEAN,
            std=image_utils.BGR_STD
        )

    def evaluate(self, outputs, batch):
        """
        Compute IOU and Rotation-Error (in degrees)
        """
        opts = self.opts
        N = batch['mask'].shape[0]

        ## compute iou
        mask_gt = batch['mask'].view(N, -1).numpy()
        mask_pred = outputs['mask_pred'].cpu().view(N, -1).type_as(
            batch['mask']).numpy()
        intersection = mask_gt * mask_pred
        union = mask_gt + mask_pred - intersection
        iou = intersection.sum(1) / union.sum(1)

        # Compute rotation error
        quat = outputs['cam_pred'][:,3:7].float().cpu()
        quat_gt = batch['sfm_pose'][:,3:7].float().cpu()
        quat_rel = geom_utils.hamilton_product(quat, geom_utils.quat_inverse(quat_gt))
        _, quat_error = geom_utils.quat2axisangle(quat_rel)
        quat_error = torch.min(quat_error, 2*np.pi - quat_error)
        quat_error = quat_error * 180 / np.pi

        # Compute x,y location of pred+gt quaternion for plotting
        azel_pred = geom_utils.camera_quat_to_position_az_el(quat, initial_quat_bias_deg=opts.initial_quat_bias_deg)
        azel_gt = geom_utils.camera_quat_to_position_az_el(quat_gt, initial_quat_bias_deg=opts.initial_quat_bias_deg)

        return iou, quat_error.numpy(), azel_pred.numpy(), azel_gt.numpy()

    def get_render_kwargs(self):
        opts = self.opts
        if opts.dataset == 'cub':
            euler_angles=[60, 90, 0]
            rot_axis=[0,-1,1.7]
            extra_elev=False
        elif opts.dataset == 'p3d':
            euler_angles = [90, 0., 0.]
            rot_axis = [0,0,1]
            extra_elev = 10
        else:
            raise ValueError()

        return {
            'euler_angles':euler_angles,
            'rot_axis':rot_axis,
            'extra_elev':extra_elev,
        }

    def render(self, outputs, batch, num_angles=6, euler_angles=[60, 90, 0], rot_axis=[0,-1,1.7], extra_elev=False):
        """
        Render the shape+texture from different viewpoints

        num_angles, euler_angles, rot_axis, extra_elev are used for rendering a 360-view
        """
        opts = self.opts
        faces = self.faces

        frame_id, img, masks = batch['inds'], batch['img'], batch['mask']
        masks = masks.unsqueeze(1)
        gt_camera_pose = batch['sfm_pose'].float().cuda()

        batch_size = len(frame_id)
        sss_batch = outputs['verts']
        ttt_batch = outputs['uv_image']
        ccc_batch = outputs['cam_pred']
        textures_nmr = self.predictor.resample_texture_nmr(ttt_batch)

        # default cameras
        euler_angles = np.array(euler_angles, dtype=np.float) * np.pi / 180
        R0 = cv2.Rodrigues(np.array([euler_angles[0], 0., 0.]))[0]
        R1 = cv2.Rodrigues(np.array([0, euler_angles[1], 0]))[0]
        R = R1.dot(R0)
        R = np.vstack((np.hstack((R, np.zeros((3, 1)))), np.array([0, 0, 0, 1])))
        rot = transformations.quaternion_from_matrix(R, isprecise=True)
        cam = np.hstack([0.75, 0, 0, rot])
        default_cam = torch.FloatTensor(cam).cuda()

        faces_batch = self.faces[None].expand(batch_size, -1, -1)
        batch_img_input = bird_vis.batchtensor2im(img)[:,:,:,::-1]
        batch_img_shape = self.predictor.vis_rend.rgba(sss_batch, cams=default_cam)
        batch_img_shape_cam = self.predictor.vis_rend.rgba(sss_batch, cams=ccc_batch)
        batch_img_shape_gtcam = self.predictor.vis_rend.rgba(sss_batch, cams=gt_camera_pose)
        batch_img_shape_cam_tex = self.predictor.vis_rend.rgba(sss_batch, cams=ccc_batch, texture=textures_nmr)[:,:,:,[2,1,0,3]]

        for _iii, (fid,gt_cam,im) in enumerate(zip(frame_id,gt_camera_pose,img)):
            ddiirr = f'{self.render_dir}/{int(fid)}/'
            if not os.path.isdir(ddiirr):
                os.mkdir(ddiirr)
            imwrite(osp.join(ddiirr, 'img_input.png'), batch_img_input[_iii])
            imwrite(osp.join(ddiirr, 'img_shape.png'), batch_img_shape[_iii])
            imwrite(osp.join(ddiirr, 'img_shape_cam.png'), batch_img_shape_cam[_iii])
            imwrite(osp.join(ddiirr, 'img_shape_gtcam.png'), batch_img_shape_gtcam[_iii])
            imwrite(osp.join(ddiirr, 'img_shape_cam_tex.png'), batch_img_shape_cam_tex[_iii])

        rend_ts = []
        rend_texs = []
        for frame_num, angle in enumerate(np.linspace(0,360,num_angles,endpoint=False)):
            rend_t = self.predictor.vis_rend.rotated(sss_batch, deg=angle, axis=rot_axis, cam=default_cam, rgba=True, extra_elev=extra_elev)  # rend_t: H,W,C
            rend_tex = self.predictor.vis_rend.rotated(sss_batch, deg=angle, axis=rot_axis, cam=default_cam, texture=textures_nmr, rgba=True, extra_elev=extra_elev)  # rend_t: H,W,C

            # rend_t = rend_t[:,:,:,[2,1,0,3]]
            rend_tex = rend_tex[:,:,:,[2,1,0,3]]
            rend_ts.append(rend_t)
            rend_texs.append(rend_tex)

            for _iii, (fid,gt_cam,im) in enumerate(zip(frame_id,gt_camera_pose,img)):
                ddiirr = f'{self.render_dir}/{int(fid)}/'
                imwrite(osp.join(ddiirr, f'img_shape_{frame_num:04d}.png'), rend_t[_iii])
                imwrite(osp.join(ddiirr, f'img_shape_tex_{frame_num:04d}.png'), rend_tex[_iii])

        for _iii, (fid,gt_cam,im) in enumerate(zip(frame_id,gt_camera_pose,img)):
            ddiirr = f'{self.render_dir}/{int(fid)}/'
            img_tile = [[batch_img_input[_iii], batch_img_shape[_iii,:,:,:3], batch_img_shape_gtcam[_iii,:,:,:3], batch_img_shape_cam[_iii,:,:,:3], batch_img_shape_cam_tex[_iii,:,:,:3]],
                [r[_iii,:,:,:3] for r in rend_ts],
                [r[_iii,:,:,:3] for r in rend_texs],
            ]
            img_tile = image_utils.concatenate_images2d(img_tile)
            imwrite(osp.join(ddiirr, 'img_tile.png'), img_tile)
            imwrite(osp.join(self.render_dir, f'{int(fid)}.png'), img_tile)



    def test(self):
        opts = self.opts
        bench_stats = {'ious': [], 'quat_error': [], 'azel_pred': [], 'azel_gt': []}

        if opts.ignore_pred_delta_v:
            result_path = osp.join(self.results_dir, 'results_meanshape.npz')
        elif opts.use_template_ms:
            result_path = osp.join(self.results_dir,
                                   'results_sfm_meanshape.npz')
        else:
            result_path = osp.join(self.results_dir, 'results.npz')

        if opts.use_gt_camera:
            result_path = result_path.replace('.npz', '_gt_camera.npz')

        print('Writing to %s' % result_path)

        if True: #not osp.exists(result_path):

            n_iter = len(self.dataloader)
            for i, batch in enumerate(tqdm(self.dataloader)):
                if i % 100 == 0:
                    print('{}/{} evaluation iterations.'.format(i, n_iter))
                if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                    break
                outputs = self.predictor.predict(batch)
                # if opts.visualize:
                #     self.visualize(outputs, batch)
                iou, quat_error, azel_pred, azel_gt = self.evaluate(outputs, batch)
                bench_stats['ious'].append(iou)
                bench_stats['quat_error'].append(quat_error)
                bench_stats['azel_pred'].append(azel_pred)
                bench_stats['azel_gt'].append(azel_gt)

                # Render images
                if opts.render:
                    self.render(outputs, batch, **self.get_render_kwargs())

                # Save shape to mat (only for p3d)
                if opts.save_mats:
                    for fid, verts in zip(batch['inds'], outputs['verts']):
                        voc_img_id = self.dataloader.dataset.anno[fid].voc_image_id
                        voc_rec_id = self.dataloader.dataset.anno[fid].voc_rec_id
                        out_dir = Path('./cachedir/evaluation/') / f'p3d_{opts.split}'
                        out_dir.mkdir(exist_ok=True, parents=True)
                        mat_file = out_dir/'{}_{}.mat'.format(voc_img_id, voc_rec_id)
                        sio.savemat(str(mat_file), {'verts': verts.detach().cpu().numpy(), 'faces': self.faces.detach().cpu().numpy()+1})

                # if opts.save_visuals and (i % opts.visuals_freq == 0):
                #     self.save_current_visuals(batch, outputs)

            bench_stats['ious'] = np.concatenate(bench_stats['ious'])
            bench_stats['quat_error'] = np.concatenate(bench_stats['quat_error'])
            bench_stats['azel_pred'] = np.concatenate(bench_stats['azel_pred'])
            bench_stats['azel_gt'] = np.concatenate(bench_stats['azel_gt'])
            np.savez(result_path, **bench_stats)
        else:
            bench_stats = np.load(result_path)

        ## Report numbers.
        mean_iou = bench_stats['ious'].mean()
        mean_qerr = bench_stats['quat_error'].mean()
        print(f'mean iou = {mean_iou:.3g}')
        print(f'mean qerr = {mean_qerr:.3g}')

        # Compute numbers: KL-divergence
        hist_kwargs = {'bins':100, 'range':[[-1.0001,1.0001]]*2, 'density':True}
        azel_hist_pred, _, _ = visutil.hist2d(bench_stats['azel_pred'], **hist_kwargs)
        azel_hist_gt, _, _ = visutil.hist2d(bench_stats['azel_gt'], **hist_kwargs)
        az_marginal = azel_hist_pred.sum(1)
        el_marginal = azel_hist_pred.sum(0)
        az_marginal_GT = azel_hist_gt.sum(1)
        el_marginal_GT = azel_hist_gt.sum(0)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(az_marginal, label=f'az_{label}')
        # plt.plot(el_marginal, label=f'el_{label}')
        # plt.plot(az_marginal_GT, label='az_GT')
        # plt.plot(el_marginal_GT, label='el_GT')
        # plt.legend()
        # plt.show()
        az_values = np.linspace(-180, 180, num=az_marginal.shape[0], endpoint=True)
        el_values = np.linspace(-90, 90, num=el_marginal.shape[0], endpoint=True)
        az_WS = scipy_stats.wasserstein_distance(az_values, az_values, az_marginal, az_marginal_GT)
        el_WS = scipy_stats.wasserstein_distance(el_values, el_values, el_marginal, el_marginal_GT)
        az_KL = scipy_stats.entropy(az_marginal, az_marginal_GT)
        el_KL = scipy_stats.entropy(el_marginal, el_marginal_GT)
        print(f'az_WS={az_WS:5.02f}  el_WS={el_WS:5.02f}')
        print(f'az_KL={az_KL:5.02f}  el_KL={el_KL:5.02f}')


        camera_azel_dict = {
            'pred': bench_stats['azel_pred'],
            'gt': bench_stats['azel_gt'],
        }

        # Scatter plots of Az-el
        for label, cam_azel in camera_azel_dict.items():
            plt.figure()
            plt.scatter(cam_azel[:,0], cam_azel[:,1], label=label, marker='.', alpha=0.6, s=5)
            plt.title(f'{label}', fontsize=13)
            plt.xlim((-1,1))
            plt.ylim((-1,1))
            plt.xlabel('azimuth', fontsize=13)
            plt.ylabel('elevation', fontsize=13)
            # plt.show()
            # plt.legend(
            #     framealpha=0.8,
            #     frameon=True,
            #     fontsize=13,
            #     facecolor='w')
            plt.draw()
            plt.savefig(f'azel_{opts.split}_{label}.png', bbox_inches='tight')


def main(_):
    opts = flags.FLAGS
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)

    # No padding/jitter while evaluating
    opts.padding_frac = 0
    opts.jitter_frac = 0

    tester = Tester(opts)
    tester.test()

if __name__ == '__main__':
    app.run(main)
