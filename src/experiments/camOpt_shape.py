from __future__ import absolute_import, division, print_function

import os

import matplotlib as mpl
from absl import app, flags

if 'DISPLAY' not in os.environ:
    print('Display not found. Using Agg backend for matplotlib')
    mpl.use('Agg')
else:
    print(f'Found display : {os.environ["DISPLAY"]}')
import itertools
import math
import os.path as osp
import pickle
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorboardX
import torch
import torchvision
from scipy.ndimage.filters import gaussian_filter

torch.backends.cudnn.benchmark = True

from ..data import cub as cub_data
from ..data import imagenet as imagenet_data
from ..data import json_dataset as json_data
from ..data import p3d as p3d_data
from ..nnutils import geom_utils, loss_utils, train_utils
from ..nnutils.architecture import ShapeCamTexNet
from ..nnutils.nmr import NeuralRenderer_pytorch as NeuralRenderer
from ..nnutils.nmr import SoftRas as SoftRas
from ..utils import bird_vis
from ..utils import image as image_utils
from ..utils import mesh
from ..utils import misc as misc_utils
from ..utils import visutil


flags.DEFINE_string('dataset', 'cub', 'yt (YouTube), or cub, or yt_filt (Youtube, refined)')
# Weights:
flags.DEFINE_float('rend_mask_loss_wt', 0, 'rendered mask loss weight')
flags.DEFINE_float('deform_loss_wt', 0, 'reg to deformation')
flags.DEFINE_float('laplacian_loss_wt', 0, 'weights to laplacian smoothness prior')
flags.DEFINE_float('meanV_laplacian_loss_wt', 0, 'weights to laplacian smoothness prior on mean shape')
flags.DEFINE_float('deltaV_laplacian_loss_wt', 0, 'weights to laplacian smoothness prior on delta shape')
flags.DEFINE_float('graphlap_loss_wt', 0, 'weights to graph laplacian smoothness prior')
flags.DEFINE_float('edge_loss_wt', 0, 'weights to edge length prior')
flags.DEFINE_float('texture_loss_wt', 0, 'weights to tex loss')
flags.DEFINE_float('camera_loss_wt', 0, 'weights to camera loss')

flags.DEFINE_boolean('perspective', False, 'whether to use strong perrspective projection')
flags.DEFINE_string('shape_path', '', 'Path to initial mean shape')
flags.DEFINE_integer('num_multipose', -1, 'num_multipose_az * num_multipose_el')
flags.DEFINE_integer('num_multipose_az', 8, 'Number of camera pose hypothesis bins (along azimuth)')
flags.DEFINE_integer('num_multipose_el', 5, 'Number of camera pose hypothesis bins (along elevation)')
flags.DEFINE_boolean('use_gt_camera', False, 'Use ground truth camera pose')
flags.DEFINE_boolean('viz_rend_video', True, 'Render video to visualize mesh')
flags.DEFINE_integer('viz_rend_steps', 36, 'Number of angles to visualize mesh from')

flags.DEFINE_float('initial_quat_bias_deg', 90, 'Rotation bias in deg. 90 for head-view, 45 for breast-view')

flags.DEFINE_float('scale_bias', 0.8, 'Scale bias for camera pose')
flags.DEFINE_boolean('optimizeCameraCont', True, 'Optimize Camera Continuously')
flags.DEFINE_float('optimizeLR', 0.0001, 'Learning rate for camera pose optimization')
flags.DEFINE_float('optimizeMomentum', 0, 'Momentum for camera pose optimization')
flags.DEFINE_enum('optimizeAlgo', 'adam', ['sgd','adam'], 'Algo for camera pose optimization')
flags.DEFINE_float('quatScorePeakiness', 20, 'quat score = e^(-peakiness * loss)')
flags.DEFINE_string('cameraPoseDict', '', 'Path to pre-computed camera pose dict for entire dataset')

flags.DEFINE_float('softras_sigma', 1e-5, 'Softras sigma (transparency)')
flags.DEFINE_float('softras_gamma', 1e-4, 'Softras gamma (blurriness)')

flags.DEFINE_boolean('texture_flipCam', False, 'Render flipped mesh and supervise using flipped image')

flags.DEFINE_boolean('pred_pose_supervise', True, 'Supervise predicted pose using best camera (in multiplex, or gt)')
flags.DEFINE_boolean('optimizeCamera_reloadCamsFromDict', False, 'Ignore camera pose from saved dictionary')

flags.DEFINE_boolean('laplacianDeltaV', False, 'Smooth DeltaV instead of V in laplacian_loss')

flags.DEFINE_float('optimizeAzRange', 30, 'Optimize Azimuth range (degrees')
flags.DEFINE_float('optimizeElRange', 30, 'Optimize Elevation range (degrees')
flags.DEFINE_float('optimizeCrRange', 60, 'Optimize CycloRotation range (degrees')

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '../../', 'cachedir')

class ForwardWrapper(torch.nn.Module):
    def __init__(self, opts, verts, faces, verts_uv, faces_uv, dataset_size):
        super().__init__()
        self.opts = opts
        self.dataset_size = dataset_size

        self.register_buffer('verts_uv', verts_uv.float())
        self.register_buffer('faces_uv', faces_uv.float())
        self.register_buffer('verts', verts.float())
        self.register_buffer('faces', faces.long())

    def define_model(self):
        opts = self.opts

        img_size = (opts.img_size, opts.img_size)
        self.model = ShapeCamTexNet(
            img_size, opts.nz_feat, opts.perspective, opts.symmetric_mesh,
            opts.pred_shape, opts.pred_pose, opts.pred_texture,
            mean_shape=self.verts, faces=self.faces,
            verts_uv_vx2=self.verts_uv, faces_uv_fx3x2=self.faces_uv,
            texture_opts=opts
        )

        if not opts.is_train:
            self.model.eval()

        self.renderer_mask = SoftRas(opts.img_size, perspective=opts.perspective, light_intensity_ambient=1.0, light_intensity_directionals=0.0)
        self.renderer_mask.ambient_light_only()
        self.renderer_mask.renderer.set_gamma(opts.softras_gamma)
        self.renderer_mask.renderer.set_sigma(opts.softras_sigma)

        _zero = torch.zeros((self.dataset_size, opts.num_multipose, 7), dtype=torch.float32)
        self.register_buffer('dataset_camera_poses', _zero)
        self.register_buffer('dataset_camera_scores', _zero[...,0].clone())
        self.register_buffer('dataset_camera_gt_st', _zero[...,0,0:3].clone())

        self.reloadCamsFromDict()

    def reloadCamsFromDict(self):
        opts = self.opts

        ## Dictionary of all camera poses
        # dict[frameid] = (Px7: [scale trans quat], P:score, 3:gtscale gttrans)
        # Store camera pose of un-flipped image only
        if opts.cameraPoseDict:
            _dd = np.load(opts.cameraPoseDict, allow_pickle=True)['campose'].item()
            print(f'Loaded cam_pose_dict of size {len(_dd)} (should be {self.dataset_size})')
            _kk = torch.tensor(list(_dd.keys()))
            assert((_kk.sort()[0]==torch.arange(_kk.shape[0])).all()) # keys are 0 -> n-1
            _dd = [_dd[k] for k in range(len(_dd))]
            dataset_camera_poses = torch.stack([c for (c,s,st) in _dd])
            dataset_camera_scores = torch.stack([s for (c,s,st) in _dd])
            dataset_camera_gt_st = torch.stack([st for (c,s,st) in _dd])
            assert(dataset_camera_poses.shape[0] == self.dataset_size)
            assert(dataset_camera_poses.shape[1] == opts.num_multipose)
        else:
            quats = geom_utils.get_base_quaternions(num_pose_az=opts.num_multipose_az,
                                                        num_pose_el=opts.num_multipose_el,
                                                        initial_quat_bias_deg=opts.initial_quat_bias_deg)
            trans = torch.zeros(quats.shape[0],2).float()
            scale = torch.zeros(quats.shape[0],1).float() + opts.scale_bias
            poses = torch.cat([scale,trans,quats], dim=-1)
            scores = torch.ones(poses.shape[0]).float()/poses.shape[0]
            gt_st = torch.tensor([opts.scale_bias, 0., 0.]).float()
            dataset_camera_poses = poses[None,...].expand(self.dataset_size, -1, -1)
            dataset_camera_scores = scores[None,...].expand(self.dataset_size, -1)
            dataset_camera_gt_st = gt_st[None,...].expand(self.dataset_size, -1)

        dataset_camera_params = torch.zeros((opts.num_multipose,1+2+3)).float()
        dataset_camera_params = [torch.nn.Parameter(dataset_camera_params.clone()) for i in range(self.dataset_size)]
        dataset_camera_params = torch.nn.ParameterList(dataset_camera_params)
        self.dataset_camera_params = dataset_camera_params

        self.dataset_camera_poses.data.copy_(dataset_camera_poses)
        self.dataset_camera_scores.data.copy_(dataset_camera_scores)
        self.dataset_camera_gt_st.data.copy_(dataset_camera_gt_st)

    def get_camera_pose(self, frame_ids, gt_st0):
        opts = self.opts

        # Flip trans in gt_st0 if frame_ids corresponds to flipped cameras
        frame_ids = frame_ids.squeeze(1)
        frame_ids_orig = torch.min(frame_ids, int(1e6)-frame_ids)
        frame_ids_flipped = (frame_ids > frame_ids_orig)
        gt_st0_flip = gt_st0 * torch.tensor([1,-1,1], dtype=gt_st0.dtype, device=gt_st0.device)
        gt_st0 = torch.where(frame_ids_flipped[:,None], gt_st0_flip, gt_st0)

        # Fetch from original
        poses = torch.index_select(self.dataset_camera_poses, 0, frame_ids_orig)    # N,P,7
        scores = torch.index_select(self.dataset_camera_scores, 0, frame_ids_orig)  # N,P
        gt_st1 = torch.index_select(self.dataset_camera_gt_st, 0, frame_ids_orig)   # N,3
        params = torch.stack([self.dataset_camera_params[i] for i in frame_ids_orig])  # N,P,6

        # Adjust for dataloader perturbation
        scale_factor = gt_st0[:,None,0:1] / gt_st1[:,None,0:1]
        scale = poses[:,:,0:1] * scale_factor
        trans = (poses[:,:,1:3] - gt_st1[:,None,1:3]) * scale_factor + gt_st0[:,None,1:3]
        quat = poses[:,:,3:7]

        # Add learnt camera_params
        scale = scale + params[:,:,0:1] *  scale_factor
        trans = trans + params[:,:,1:3] *  scale_factor
        az = torch.tanh(params[:,:,3:4]) * np.pi * opts.optimizeAzRange/180   # max 30 deg
        el = torch.tanh(params[:,:,4:5]) * np.pi * opts.optimizeElRange/180   # max 30 deg
        cr = torch.tanh(params[:,:,5:6]) * np.pi * opts.optimizeCrRange/180   # max 60 deg
        azelcr = torch.cat([az, el, cr], dim=-1)
        quat2 = geom_utils.azElRot_to_quat(azelcr)
        quat = geom_utils.hamilton_product(quat2, quat)

        # Flip camera pose if frame_ids_flipped
        camera_pose = torch.cat((scale, trans, quat), dim=-1) # N,P,7
        camera_pose_ref = geom_utils.reflect_cam_pose(camera_pose)
        camera_pose_fin = torch.where(frame_ids_flipped[:,None,None], camera_pose_ref, camera_pose)
        return camera_pose_fin, scores

    def define_criterion(self):
        self.rend_mask_loss_fn = loss_utils.mask_l2_dt_loss

        # For shape
        self.deform_loss_fn = loss_utils.deform_l2reg
        self.laplacian_loss_fn = loss_utils.LaplacianLoss(self.faces.long(), self.verts.detach())
        self.meanV_laplacian_loss_fn = loss_utils.LaplacianLoss(self.faces.long(), self.verts.detach())
        self.deltaV_laplacian_loss_fn = loss_utils.LaplacianLoss(self.faces.long(), self.verts.detach())
        self.graphlap_loss_fn = loss_utils.GraphLaplacianLoss(self.faces.long(), self.verts.shape[0])
        self.edge_loss_fn = loss_utils.EdgeLoss(self.verts.detach(), self.faces.long())

        # For texture
        if self.opts.pred_texture:
            self.add_module('texture_loss_fn', loss_utils.PerceptualTextureLoss())

    def forward(self, input_dict):
        """
        input dict = {
            img: normalized images
            mask: silhouette
            gt_camera_pose: (optional) GT camera pose

            flip_img: flipped normalized images
            flip_mask: flipped silhouette
            flip_gt_camera_pose: (optional) flipped GT camera pose
        }
        """
        opts = self.opts
        self.real_iter = input_dict['real_iter']
        self.epochs_done = input_dict['epochs_done']

        assert(not opts.flip_train)
        self.orig_img = input_dict['orig_img']
        self.input_img = input_dict['img']
        self.input_mask = input_dict['mask']
        self.input_mask_dt = input_dict['mask_dt']
        self.frame_id = input_dict['frame_id']

        self.gt_camera_pose = input_dict['gt_camera_pose']

        P = opts.num_multipose
        N, h, w = self.input_mask.shape
        assert(self.input_img.shape[0] == N)
        assert(self.input_img.shape[2:] == (h, w))

        # Expand mask for multi_pose
        self.mask = self.input_mask[:,None,:,:].repeat(1,P,1,1)
        self.mask = self.mask.view(N * P, h, w)
        self.mask_dt = self.input_mask_dt[:,None,:,:].repeat(1,P,1,1)
        self.mask_dt = self.mask_dt.view(N * P, h, w)

        ## PREDICT SHAPE, etc.
        delta_v, textures, pred_cam = self.model(self.input_img)

        if opts.use_gt_camera:
            assert(P==1)
            self.cam_poses = self.gt_camera_pose[:,None,:]
        else:
            self.cam_poses, self.cam_scores = self.get_camera_pose(self.frame_id, self.gt_camera_pose[...,0:3])

        NUM_BATCH_POSE = N*P
        delta_v_orig = delta_v
        delta_v = delta_v[:,None,...].repeat(1,P,1,1)
        delta_v = delta_v.view((NUM_BATCH_POSE,) + delta_v.shape[2:])
        cam_pred = self.cam_poses.view((NUM_BATCH_POSE,) + self.cam_poses.shape[2:])

        # Deform mean shape:
        mean_shape = self.model.get_mean_shape()
        pred_v = mean_shape[None,:,:] + (delta_v)

        # Texture stuff: texture_flow, textures
        if opts.pred_texture:
            tex_size = textures.size(2)
            textures = textures.unsqueeze(4).expand(-1, -1, -1, -1, tex_size, -1) # B,F,T,T,T,3
            textures_rep = textures[:,None,...].repeat(1,P,1,1,1,1,1)
            textures_rep = textures_rep.view((NUM_BATCH_POSE,) + textures_rep.shape[2:])
            self.textures = textures.detach()
            self.texture_uvimage_pred = self.model.texturePred.uvimage_pred.detach()
        else:
            self.textures = None
            self.texture_uvimage_pred = None


        # Render mask, texture
        faces_batch = self.faces[None,:,:].expand(pred_v.shape[0],-1,-1)
        if opts.pred_texture:
            rend_texture, rend_mask = self.renderer_mask.render_texture_mask(pred_v, faces_batch.int(), cam_pred, textures=textures_rep)
            if opts.texture_flipCam:
                _cam = geom_utils.reflect_cam_pose(cam_pred)
                rend_texture_flip, _ = self.renderer_mask.render_texture_mask(pred_v, faces_batch.int(), _cam, textures=textures_rep)
        else:
            rend_mask = self.renderer_mask.forward(pred_v, faces_batch.int(), cam_pred)
            rend_texture = None

        if opts.pred_texture:
            imgs_batch = self.orig_img[:,None].repeat(1,P,1,1,1)
            imgs_batch = imgs_batch.view((NUM_BATCH_POSE,)+imgs_batch.shape[2:])
            self.texture_loss_mp = self.texture_loss_fn(rend_texture, imgs_batch, rend_mask, self.mask)
            if opts.texture_flipCam:
                imgs_batch_flip = torch.flip(imgs_batch, [-1])
                mask_flip = torch.flip(self.mask, [-1])
                texture_flip_loss_mp = self.texture_loss_fn(rend_texture_flip, imgs_batch_flip, None, mask_flip)
                self.texture_loss_mp = (self.texture_loss_mp + texture_flip_loss_mp)/2  # TODO: FIX

            self.texture_loss_mp = self.texture_loss_mp.view(N, P)
        else:
            self.texture_loss_mp = torch.zeros((N, P), dtype=torch.float32, device=self.input_img.device)

        assert(torch.isfinite(mean_shape).all())
        assert(torch.isfinite(pred_v).all())
        assert(torch.isfinite(delta_v).all())
        assert(torch.isfinite(cam_pred).all())

        # Calculate optimized camera quaternion error
        quat = cam_pred[:,3:7].view(N,  P, 4)
        quat_error = geom_utils.hamilton_product(quat, geom_utils.quat_inverse(self.gt_camera_pose[:,None,3:7]))
        _, quat_error = geom_utils.quat2axisangle(quat_error)
        quat_error = torch.min(quat_error, 2*np.pi - quat_error)
        self.quat_error = quat_error * 180 / np.pi

        # Calculate network-predicted quaternion error
        if opts.pred_pose:
            quat = pred_cam[:,3:7]
            quat_error = geom_utils.hamilton_product(quat, geom_utils.quat_inverse(self.gt_camera_pose[:,3:7]))
            _, quat_error = geom_utils.quat2axisangle(quat_error)
            quat_error = torch.min(quat_error, 2*np.pi - quat_error)
            self.network_quat_error = quat_error * 180 / np.pi
        else:
            self.network_quat_error = self.quat_error[:,0]

        ### Loss computation ###
        ## Rend mask loss
        self.rend_mask_loss_mp = self.rend_mask_loss_fn(rend_mask, self.mask, reduction='none', mask2_dt=self.mask_dt) \
                                                .mean(dim=(-1,-2))                              \
                                                .view(N, P)
        self.rend_mask_iou_mp = loss_utils.maskiou(rend_mask, self.mask) \
                                                .view(N, P)

        # Per-camera loss for computing camera weights
        if opts.pred_texture:
            _mask_loss = self.rend_mask_loss_mp
            _mask_loss_min, _ = _mask_loss.min(dim=1, keepdim=True)
            _mask_loss_max, _ = _mask_loss.max(dim=1, keepdim=True)
            _mask_loss_rescaled = (_mask_loss-_mask_loss_min)/(_mask_loss_max-_mask_loss_min + 1e-4)

            _texture_loss = self.texture_loss_mp
            _texture_loss_min, _ = _texture_loss.min(dim=1, keepdim=True)
            _texture_loss_max, _ = _texture_loss.max(dim=1, keepdim=True)
            _texture_loss_rescaled = (_texture_loss-_texture_loss_min)/(_texture_loss_max-_texture_loss_min + 1e-4)

            rend_weight = 0.5
            texture_weight = 0.5

            self.total_loss_mp = rend_weight*_mask_loss_rescaled + texture_weight*_texture_loss_rescaled
        else:
            self.total_loss_mp = opts.rend_mask_loss_wt * self.rend_mask_loss_mp


        # # Score from total_loss_mp
        cam_pose_loss = self.total_loss_mp.detach()
        loss_min,_ = cam_pose_loss.min(dim=1)
        loss_max,_ = cam_pose_loss.max(dim=1)
        loss_rescaled = (cam_pose_loss - loss_min[:,None])/(loss_max[:,None] - loss_min[:,None] + 1e-6)
        self.quat_score = torch.nn.functional.softmin(loss_rescaled * opts.quatScorePeakiness, dim=1)

        if opts.pred_pose and opts.pred_pose_supervise:
            _mm, _ii = self.quat_score.max(dim=1)
            _rr = torch.arange(_ii.shape[0], dtype=_ii.dtype, device=_ii.device)
            _gt_cam_bx7 = self.cam_poses[_rr, _ii, :]
            self.camera_loss = loss_utils.camera_loss(pred_cam, _gt_cam_bx7, 0)
        else:
            self.camera_loss = torch.tensor(0., device=self.input_img.device)

        self.rend_mask_loss = (self.rend_mask_loss_mp*self.quat_score).sum(dim=1).mean()
        self.texture_loss = (self.texture_loss_mp*self.quat_score).sum(dim=1).mean()

        # Shape priors:
        _zero = torch.tensor(0., device=self.input_img.device)
        self.deform_loss = self.deform_loss_fn(delta_v)
        self.laplacian_loss = self.laplacian_loss_fn(delta_v if opts.laplacianDeltaV else pred_v) if opts.laplacian_loss_wt>0 else _zero
        self.meanV_laplacian_loss = self.meanV_laplacian_loss_fn(mean_shape[None,:,:]) if opts.meanV_laplacian_loss_wt>0 else _zero
        self.deltaV_laplacian_loss = self.deltaV_laplacian_loss_fn(delta_v) if opts.deltaV_laplacian_loss_wt>0 else _zero
        self.graphlap_loss = self.graphlap_loss_fn(pred_v) if opts.graphlap_loss_wt>0 else _zero
        self.edge_loss = self.edge_loss_fn(pred_v) if opts.edge_loss_wt>0 else _zero

        # Loss for camera pose update
        self.camera_update_loss = opts.rend_mask_loss_wt * self.rend_mask_loss_mp.mean() + \
                                    opts.texture_loss_wt * self.texture_loss_mp.mean()

        # Finally sum up the loss.
        self.total_loss = 0
        # Shape Priors:
        self.total_loss += opts.edge_loss_wt * self.edge_loss
        self.total_loss += opts.laplacian_loss_wt * self.laplacian_loss
        self.total_loss += opts.meanV_laplacian_loss_wt * self.meanV_laplacian_loss
        self.total_loss += opts.deltaV_laplacian_loss_wt * self.deltaV_laplacian_loss
        self.total_loss += opts.graphlap_loss_wt * self.graphlap_loss
        self.total_loss += opts.deform_loss_wt * self.deform_loss
        # Instance loss:
        self.total_loss += opts.rend_mask_loss_wt * self.rend_mask_loss
        self.total_loss += opts.texture_loss_wt * self.texture_loss
        # self.total_loss += opts.texture_dt_loss_wt * self.texture_dt_loss
        self.total_loss += opts.camera_loss_wt * self.camera_loss

        ### Save variables in natural dimensions for visualization
        self.delta_v = delta_v_orig
        self.cam_pred = cam_pred.view(N, P,7)
        self.pred_v = pred_v.view((N, P,)+pred_v.shape[1:])[:,0,:,:]
        self.mean_shape = mean_shape
        self.rend_texture = rend_texture.view((N, P)+rend_texture.shape[1:])

        ### Statistics
        self.statistics = {}
        if opts.pred_pose:
            self.statistics['network_quat_values'] = pred_cam[...,3:7]  # N,P,4
            self.statistics['network_quat_error'] = self.network_quat_error # N,P
            self.statistics['network_cam_values'] = pred_cam  # N,P,4
        else:
            self.statistics['network_quat_values'] = self.cam_pred[:,0,3:7]   # N,P,4
            self.statistics['network_quat_error'] = self.quat_score[:,0]    # N,P
            self.statistics['network_cam_values'] = self.cam_pred[:,0,:]   # N,P,4
        self.statistics['quat_values'] = self.cam_pred[...,3:7] # N,P,4
        self.statistics['cam_values'] = self.cam_pred          # N,P,4
        self.statistics['quat_scores'] = self.quat_score        # N,P
        self.statistics['gt_quat'] = self.gt_camera_pose[:,3:7] # N,4
        self.statistics['gt_cam'] = self.gt_camera_pose[:,:] # N,4
        self.statistics['quat_error'] = self.quat_error         # N,P
        self.statistics['rend_mask_iou_mp'] = self.rend_mask_iou_mp         # N,P
        self.statistics['shape'] = torch.tensor([0.])
        self.statistics['texture_uv'] = torch.tensor([0.])

        self.total_loss_num = torch.zeros_like(self.total_loss) + len(self.input_img)
        return_attr = {
            ## Inputs
            'input_img',
            'input_mask',
            'frame_id',
            'gt_camera_pose',

            ## Losses
            'camera_update_loss',
            'total_loss_num',
            'total_loss',
            'total_loss_mp',
            'rend_mask_loss_mp',
            'rend_mask_iou_mp',
            'edge_loss',
            'laplacian_loss',
            'meanV_laplacian_loss',
            'deltaV_laplacian_loss',
            'graphlap_loss',
            'deform_loss',
            'rend_mask_loss',
            'texture_loss',
            'texture_loss_mp',
            'camera_loss',

            ## Intermediate values
            'quat_score',
            'pred_v',
            'cam_pred',
            'mean_shape',
            'quat_error',
            'network_quat_error',
            'textures',
            'texture_uvimage_pred',
            'rend_texture',

            # Statistics
            'statistics',
        }
        return_dict = {
            k:getattr(self,k) for k in return_attr
        }
        return return_dict

class ShapeTrainer(train_utils.Trainer):
    def define_model(self):
        opts = self.opts

        #######################
        ### Setup Mean Shape
        #######################
        mean_shape = mesh.fetch_mean_shape(opts.shape_path, mean_centre_vertices=True)
        verts = mean_shape['verts']
        faces = mean_shape['faces']
        verts_uv = mean_shape['verts_uv']
        faces_uv = mean_shape['faces_uv']

        # # Visualize uvmap
        # misc_utils.plot_triangles(faces_uv)

        self.verts_uv = torch.from_numpy(verts_uv).float() # V,2
        self.verts = torch.from_numpy(verts).float() # V,3
        self.faces = torch.from_numpy(faces).long()  # F,2
        self.faces_uv = torch.from_numpy(faces_uv).float()  # F,3,2

        assert(verts_uv.shape[0] == verts.shape[0])
        assert(verts_uv.shape[1] == 2)
        assert(verts.shape[1] == 3)
        assert(faces.shape[1] == 3)
        assert(faces_uv.shape == (faces.shape)+(2,))

        # Store UV sperical texture map
        verts_sph = geom_utils.convert_uv_to_3d_coordinates(verts_uv)
        if not opts.textureUnwrapUV:
            uv_sampler = mesh.compute_uvsampler_softras(verts_sph, faces, tex_size=opts.tex_size, shift_uv=opts.texture_uvshift)
        else:
            uv_sampler = mesh.compute_uvsampler_softras_unwrapUV(faces_uv, faces, tex_size=opts.tex_size, shift_uv=opts.texture_uvshift)

        uv_texture = visutil.uv2bgr(uv_sampler) # F,T,T,3
        uv_texture = np.repeat(uv_texture[:,:,:,None,:], opts.tex_size, axis=3) # F,T,T,T,2
        self.uv_texture = torch.tensor(uv_texture).float().cuda()/255.

        if not opts.textureUnwrapUV:
            uv_sampler_nmr = mesh.compute_uvsampler(verts_sph, faces, tex_size=opts.tex_size, shift_uv=opts.texture_uvshift)
        else:
            uv_sampler_nmr = mesh.compute_uvsampler_unwrapUV(faces_uv, faces, tex_size=opts.tex_size, shift_uv=opts.texture_uvshift)
        self.uv_sampler_nmr = torch.tensor(uv_sampler_nmr).float().cuda()

        # Store UV-colors of vertices
        verts_BGR = visutil.uv2bgr(self.verts_uv.detach().cpu().numpy()) # N,3
        self.verts_BGR = torch.tensor(verts_BGR).float().cuda()/255.

        # Renderer for visualization
        self.vis_rend = bird_vis.VisRenderer(opts.img_size, faces, perspective=opts.perspective)


        ####################
        # Define model
        ####################
        self.model = ForwardWrapper(opts, self.verts, self.faces, self.verts_uv, self.faces_uv, len(self.dataloader.dataset))
        self.model.define_model()
        self.model.define_criterion()
        if opts.pretrained_epoch_label or opts.pretrained_network_path:
            self.load_network(self.model.model, 'pred', opts.pretrained_epoch_label, network_dir=opts.pretrained_network_dir, path=opts.pretrained_network_path)

        if (opts.pretrained_camera_params_path or opts.pretrained_epoch_label) and (not opts.use_gt_camera):
            if opts.pretrained_camera_params_path:
                path = opts.pretrained_camera_params_path
            elif opts.pretrained_epoch_label:
                if opts.pretrained_network_dir == '':
                    network_dir = self.save_dir
                else:
                    network_dir = opts.pretrained_network_dir
                save_filename = 'camera_params_{}.pth'.format(opts.pretrained_epoch_label)
                path = os.path.join(network_dir, save_filename)
            data = torch.load(path)
            assert(len(data) == len(self.model.dataset_camera_params))
            for i in range((len(data))):
                self.model.dataset_camera_params[i].data.copy_(data[i].data)

        if opts.optimizeCamera_reloadCamsFromDict:
            self.model.reloadCamsFromDict()

        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

        return

    def init_dataset(self):
        opts = self.opts
        if opts.dataset == 'cub':
            dataloader_fn = cub_data.data_loader
        elif opts.dataset == 'imnet':
            dataloader_fn = imagenet_data.imnet_dataloader
        elif opts.dataset == 'p3d':
            dataloader_fn = p3d_data.data_loader
        elif opts.dataset == 'json':
            dataloader_fn = json_data.data_loader
        else:
            raise ValueError('Unknown dataset %d!' % opts.dataset)

        self.dataloader = dataloader_fn(opts)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=torch.tensor(image_utils.BGR_MEAN, dtype=torch.float),
            std=torch.tensor(image_utils.BGR_STD, dtype=torch.float)
        )

    def define_criterion(self):
        pass

    def set_input(self, batch):
        opts = self.opts

        # Batch
        frame_id, img, masks, masks_dt = batch['inds'], batch['img'], batch['mask'], batch['mask_dt']

        # GT camera pose (from sfm_pose) for debugging
        gt_camera_pose = batch['sfm_pose'].float()

        N, h, w = masks.shape
        assert(img.shape[0] == N)
        assert(img.shape[2:] == (h, w))
        self.N = N
        self.H = h
        self.W = w

        img = img.float()
        masks = masks.float()
        masks_dt = masks_dt.float()

        input_img = img.clone()
        for b in range(input_img.size(0)):
            input_img[b] = self.resnet_transform(input_img[b])

        self.frame_id = frame_id
        self.img = img.cuda(non_blocking=True)
        self.input_img = input_img.cuda(non_blocking=True)
        self.mask = masks.cuda(non_blocking=True)
        self.mask_dt = masks_dt.cuda(non_blocking=True)
        self.gt_camera_pose = gt_camera_pose.cuda(non_blocking=True)

    def forward(self, do_optimize=False, **kwargs):
        opts = self.opts

        input_dict = {
            'orig_img':self.img,
            'img':self.input_img,
            'mask':self.mask,
            'mask_dt':self.mask_dt,
            'real_iter':self.real_iter,
            'epochs_done':self.epochs_done,
            'frame_id':self.frame_id,
            'do_optimize':do_optimize,
            'gt_camera_pose':self.gt_camera_pose,
        }

        self.return_dict = self.model(input_dict)
        loss_keys = {
            # Losses
            'camera_update_loss',
            'total_loss',
            'edge_loss',
            'laplacian_loss',
            'meanV_laplacian_loss',
            'deltaV_laplacian_loss',
            'graphlap_loss',
            'deform_loss',
            'rend_mask_loss',
            'texture_loss',
            'camera_loss',
        }
        gpu_weights = self.return_dict['total_loss_num']/self.return_dict['total_loss_num'].sum()
        for k in loss_keys:
            setattr(self,k,(self.return_dict[k] * gpu_weights).mean())
        self.update_epoch_statistics()

        cam_pred = self.return_dict['cam_pred']
        gt_st = self.return_dict['gt_camera_pose'][...,0:3] # scale trans
        frame_id = self.return_dict['frame_id']
        quat_score = self.return_dict['quat_score']
        maskiou = self.return_dict['rend_mask_iou_mp']

        if do_optimize:
            assert(False)

        # if opts.save_camera_pose_dict:
        self.update_camera_pose_dict(frame_id.squeeze(1), cam_pred, quat_score, gt_st, maskiou)

    def update_camera_pose_dict(self, frameids, cams, scores, gt_st, maskiou):
        ## Dictionary of all camera poses
        # dict[frameid] = (Px7: [scale trans quat], P:score, 3:gtscale gttrans)
        assert(frameids.shape[0] == cams.shape[0] == scores.shape[0] == gt_st.shape[0] == maskiou.shape[0])
        assert(frameids.dim()==1)
        assert(scores.dim()==2)
        assert(gt_st.dim()==2)
        assert(cams.dim()==3)

        frameids = frameids.detach().cpu()
        cams = cams.detach().cpu()
        scores = scores.detach().cpu()
        gt_st = gt_st.detach().cpu()
        maskiou = maskiou.detach().cpu()


        frame_id_isflip = frameids > (int(1e6)-frameids)
        flip_cams = geom_utils.reflect_cam_pose(cams)
        flip_gt_st = gt_st * torch.tensor([1,-1,1], dtype=gt_st.dtype, device=gt_st.device)
        gt_st = torch.where(frame_id_isflip[:,None], flip_gt_st, gt_st)
        cams = torch.where(frame_id_isflip[:,None,None], flip_cams, cams)
        frameids = torch.where(frame_id_isflip, int(1e6)-frameids, frameids)

        for i in range(frameids.shape[0]):
            f = frameids[i].item()
            if f not in self.datasetCameraPoseDict:
                self.datasetCameraPoseDict[f] = (cams[i,:,:], scores[i,:], gt_st[i,:])

    def get_current_visuals(self):
        vis_dict = {'img':{}, 'mesh':{}, 'video':{}, 'text':{}}

        opts = self.opts

        quat_score = self.return_dict['quat_score']
        pred_v = self.return_dict['pred_v']
        cam_pred = self.return_dict['cam_pred']
        total_loss_mp = self.return_dict['total_loss_mp'].cpu()
        rend_mask_loss_mp = self.return_dict['rend_mask_loss_mp'].cpu() * opts.rend_mask_loss_wt
        rend_mask_iou_mp = self.return_dict['rend_mask_iou_mp'].cpu()
        quat_error = self.return_dict['quat_error']
        mean_shape = self.return_dict['mean_shape']
        if len(mean_shape.shape) > 2:
            mean_shape = mean_shape[0]

        images = self.return_dict['input_img']
        img_mask = self.return_dict['input_mask']
        frame_id = self.return_dict['frame_id']

        images_gpu = image_utils.unnormalize_img(images.detach())
        images = images_gpu.cpu()
        img_mask = img_mask.detach().cpu()
        frame_id = frame_id.detach().cpu()

        quat_score = quat_score.detach().cpu()
        pred_v = pred_v.detach()
        cam_pred = cam_pred.detach()

        if opts.pred_texture:
            rend_texture = self.return_dict['rend_texture'].detach().cpu()
            textures = self.return_dict['textures'].detach()
            uv_images = self.return_dict['texture_uvimage_pred'].detach()
            texture_loss_mp = self.return_dict['texture_loss_mp'].cpu() * opts.texture_loss_wt

            uv_sampler_fxttx2 = self.uv_sampler_nmr.view(self.faces.shape[0], -1, 2)
            uv_sampler_bxfxttx2 = uv_sampler_fxttx2[None].expand(uv_images.shape[0], -1, -1, -1)
            tex_pred_bx3xfxtt = torch.nn.functional.grid_sample(uv_images, uv_sampler_bxfxttx2)
            tex_pred_bxfxtxtx3 = tex_pred_bx3xfxtt.view(uv_images.shape[0], 3, self.faces.shape[0], opts.tex_size, opts.tex_size).permute(0, 2, 3, 4, 1)
            textures_nmr = tex_pred_bxfxtxtx3.unsqueeze(4).expand(-1, -1, -1, -1, opts.tex_size, -1)

            uv_images = uv_images.detach().cpu()

        num_show = min(8, images.shape[0])
        for i in range(num_show):
            img = bird_vis.tensor2im(images[i])
            img_overlay = bird_vis.tensor2im(images[i])
            # img_overlay_chckbrd = images[i]*0.1 + self.chckbrd_img.cpu()*0.9

            # Testing camera augmentation
            _, min_j = total_loss_mp[i,:].min(dim=0)
            pred_v_ij = self.vis_rend.renderer.project_points_perspective(pred_v[None,i], cam_pred[None,i,min_j.item()], depth=False).squeeze(0).cpu() # N,2
            for vi in range(pred_v_ij.shape[0]):
                xx = (pred_v_ij[vi,0]+1)/2 * img.shape[0]
                yy = (pred_v_ij[vi,1]+1)/2 * img.shape[0]
                img_overlay = cv2.circle(img_overlay,(xx,yy), 1, (0,255,0), -1)

            mask = bird_vis.tensor2mask(img_mask[i])
            plot_imgs = [[img,mask]]

            viewpoints_imgs = []
            viewpoints_texture = []
            _, min_j = total_loss_mp[i,:].min(dim=0)
            for j in range(opts.num_multipose):
                rend_predcam = self.vis_rend(pred_v[i], cam_pred[i,j], texture=self.uv_texture)

                if opts.pred_texture:
                    self.vis_rend.set_ambient(True)
                    self.vis_rend.set_ambient(False)
                    rend_texture_ij = bird_vis.tensor2im(rend_texture[i,j])
                    viewpoints_texture.append(rend_texture_ij)

                text_color = (0,255,0) if (min_j==j) else (255,255,255)
                text_img = rend_predcam
                texts = [
                    f'c{j} {quat_score[i,j]*100:.01f}%',
                    f'L {total_loss_mp[i,j]:.03f}',
                    f'RL {rend_mask_loss_mp[i,j]:.03f}',
                    f'iou {rend_mask_iou_mp[i,j]:.03f}',
                    f'TL {texture_loss_mp[i,j]:.03f}',
                    f'E {quat_error[i,j]:.01f}d',
                ]
                for _i,_t in enumerate(texts):
                    text_img = cv2.putText(text_img,_t,(5,20*(_i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,text_color,1,cv2.LINE_AA)
                viewpoints_imgs.append(text_img)

            def tile_list(_l, num_per_row=opts.num_multipose_az):
                _ll = []
                for i in range(math.ceil(len(_l)/num_per_row)):
                    _ll.append(_l[i*num_per_row : (i+1)*num_per_row])
                return _ll

            plot_imgs = tile_list(viewpoints_imgs)
            if opts.pred_texture:
                plot_texture = tile_list(viewpoints_texture)

            tag_prefix = i

            vis_dict['img']['%d/0' % tag_prefix] = image_utils.concatenate_images2d(plot_imgs)[:,:,::-1]
            best_imgs_list = [[img,img_overlay,mask,viewpoints_imgs[min_j]]]

            if opts.pred_texture:
                vis_dict['img']['%d/0_uvimg' % tag_prefix] = bird_vis.tensor2im(uv_images[i])[:,:,::-1]
                vis_dict['img']['%d/0_texture' % tag_prefix] = image_utils.concatenate_images2d(plot_texture)[:,:,::-1]
                best_imgs_list_row = []
                best_imgs_list_row.append(viewpoints_texture[min_j])
                best_imgs_list.append(best_imgs_list_row)

            vis_dict['img']['%d/0_best' % tag_prefix] = image_utils.concatenate_images2d(best_imgs_list)[:,:,::-1]

            vis_dict['text']['%d/0' % tag_prefix] = f'{frame_id[i].item()}'

            if opts.viz_rend_video:
                rends = []
                rends_tex = []
                for angle in np.linspace(0,360,opts.viz_rend_steps,endpoint=False):
                    if opts.dataset in ['cub']:
                        default_cam = None
                        axis=[0,-1,1.7]
                    else:
                        default_cam = torch.tensor([0.5,0,0,1,1,0,0], dtype=torch.float, device=self.vis_rend.default_cam.device)
                        axis=[0,0,1.7]
                    rend_t = self.vis_rend.rotated(pred_v[i], deg=angle, axis=axis, cam=default_cam)  # rend_t: H,W,C
                    if opts.pred_texture:
                        rend_tex = self.vis_rend.rotated(pred_v[i], deg=angle, axis=axis, cam=default_cam, texture=textures_nmr[i])  # rend_t: H,W,C
                    rends.append(rend_t)
                    if opts.pred_texture:
                        rend_tex = rend_tex[:,:,::-1]
                        rends_tex.append(rend_tex)
                rends = np.stack(rends, axis=0)                             # rends: t,H,W,C
                if opts.pred_texture:
                    rends_tex = np.stack(rends_tex, axis=0)                             # rends: t,H,W,C
                    vis_dict['video']['%d/0_texmesh' % tag_prefix] = rends_tex.transpose((0,3,1,2))  # t,C,H,W
                vis_dict['video']['%d/0_mesh' % tag_prefix] = rends.transpose((0,3,1,2))  # t,C,H,W
                vis_dict['video_fps'] = opts.viz_rend_steps / 12        # see each frame for 12 seconds

        lights = [
            {
            'cls': 'AmbientLight',
            'color': '#ffffff',
            'intensity': 0.75,
            }, {
            'cls': 'DirectionalLight',
            'color': '#ffffff',
            'intensity': 0.75,
            'position': [0, -1, 2],
            },{
            'cls': 'DirectionalLight',
            'color': '#ffffff',
            'intensity': 0.75,
            'position': [5, 1, -2],
            },
        ]
        material = {'cls': 'MeshStandardMaterial', 'side': 2}
        config = {'material':material, 'lights':lights}

        faces_batch = self.faces[None,...].expand(pred_v.shape[0],-1,-1)

        mean_shape = mean_shape.unsqueeze(0).detach().cpu().contiguous()
        vis_dict['mesh']['mean_shape'] = {'v': mean_shape, 'f': self.faces[None,...].detach().cpu().contiguous(), 'cfg':config,}

        pred_v = pred_v.detach().cpu().contiguous()
        faces_batch = faces_batch.detach().cpu().contiguous()
        for i in range(num_show):
            tag_prefix = i
            vis_dict['mesh'][f'{tag_prefix}'] = {'v': pred_v[i:i+1], 'f': faces_batch[i:i+1], 'cfg':config}


        return vis_dict

    def get_current_scalars(self):
        loss_values = [
            'rend_mask_loss',
            'edge_loss',
            'laplacian_loss',
            'meanV_laplacian_loss',
            'deltaV_laplacian_loss',
            'graphlap_loss',
            'deform_loss',
            'texture_loss',
            'camera_loss',
        ]
        weighted_loss = {
            attr:getattr(self.opts,f'{attr}_wt') * getattr(self,attr).item() for attr in loss_values
        }
        percent_contrib = {
            attr:(getattr(self.opts,f'{attr}_wt') * getattr(self,attr).item())/self.total_loss.item() for attr in loss_values
        }
        sc_dict = [
            ('smoothed_total_loss', self.smoothed_total_loss),
            ('total_loss', self.total_loss.item()),
            ('weighted_loss',weighted_loss),
            ('percent_contrib',percent_contrib),
        ]
        for attr in loss_values:
            sc_dict.append((attr,getattr(self, attr).item()))
        sc_dict = OrderedDict(sc_dict)
        return sc_dict

    def save(self, epoch_prefix):
        '''Saves the model.'''
        self.save_network(self.model.module.model, 'pred', epoch_prefix)
        self.save_network(self.model.module.model.convEncoder, 'convnet', epoch_prefix)
        self.save_network(self.model.module.model.posePred, 'posenet', epoch_prefix)
        self.save_network(self.model.module.model.shapePred, 'shapenet', epoch_prefix)
        self.save_network(self.model.module.model.texturePred, 'texnet', epoch_prefix)

        # Save camera parameters
        save_filename = 'camera_params_{}.pth'.format(epoch_prefix)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.model.module.dataset_camera_params, save_path)
        return

    def reset_epoch_statistics(self):
        self.epoch_statistics = {
            'gt_quat':[],               # N,4    gt quat
            'gt_cam':[],                # N,4    gt quat
            'network_quat_values':[],   # N,P,4  net-predicted camera pose
            'network_cam_values':[],    # N,P,4  net-predicted camera pose
            'quat_values':[],           # N,P,4  predicted camera pose
            'cam_values':[],            # N,P,4  predicted camera pose
            'quat_scores':[],           # N,P    camera pose scores
            'quat_error':[],            # N,P    quat error in degrees
            'network_quat_error':[],    # N,P    net-predicted quat error in degrees
            'input_img':[],             # N,3,h,w input images
            'frame_id':[],              # N,     frame ids
            'rend_mask_iou_mp':[],      # N,P    quat error in degrees
            'shape':[],                 # N,P,4  net-predicted camera pose
            'texture_uv':[],            # N,P,4  net-predicted camera pose

        }
        ## Dictionary of all camera poses
        # dict[frameid] = (Px7: [scale trans quat], P:score, 3:gtscale gttrans)
        self.datasetCameraPoseDict = {}

    def update_epoch_statistics(self):
        statistics = self.return_dict['statistics']
        self.epoch_statistics['gt_quat'].append(statistics['gt_quat'].detach().cpu())
        self.epoch_statistics['gt_cam'].append(statistics['gt_cam'].detach().cpu())
        self.epoch_statistics['network_quat_values'].append(statistics['network_quat_values'].detach().cpu())
        self.epoch_statistics['network_cam_values'].append(statistics['network_cam_values'].detach().cpu())
        self.epoch_statistics['quat_values'].append(statistics['quat_values'].detach().cpu())
        self.epoch_statistics['cam_values'].append(statistics['cam_values'].detach().cpu())
        self.epoch_statistics['quat_scores'].append(statistics['quat_scores'].detach().cpu())
        self.epoch_statistics['quat_error'].append(statistics['quat_error'].detach().cpu())
        self.epoch_statistics['network_quat_error'].append(statistics['network_quat_error'].detach().cpu())
        self.epoch_statistics['rend_mask_iou_mp'].append(statistics['rend_mask_iou_mp'].detach().cpu())
        self.epoch_statistics['input_img'].append(self.return_dict['input_img'].detach().cpu())
        self.epoch_statistics['frame_id'].append(self.return_dict['frame_id'].detach().cpu())
        self.epoch_statistics['shape'].append(statistics['shape'].detach().cpu())
        self.epoch_statistics['texture_uv'].append(statistics['texture_uv'].detach().cpu())

    def get_epoch_statistics(self):
        gt_quat = torch.cat(self.epoch_statistics['gt_quat'], dim=0)     # N,4
        gt_cam = torch.cat(self.epoch_statistics['gt_cam'], dim=0)     # N,4
        network_quat_values = torch.cat(self.epoch_statistics['network_quat_values'], dim=0) # N,P,4
        network_cam_values = torch.cat(self.epoch_statistics['network_cam_values'], dim=0) # N,P,4
        quat_values = torch.cat(self.epoch_statistics['quat_values'], dim=0) # N,P,4
        cam_values = torch.cat(self.epoch_statistics['cam_values'], dim=0) # N,P,4
        quat_scores = torch.cat(self.epoch_statistics['quat_scores'], dim=0) # N,P
        quat_error = torch.cat(self.epoch_statistics['quat_error'], dim=0)  # N,P
        network_quat_error = torch.cat(self.epoch_statistics['network_quat_error'], dim=0)  # N,P
        frame_id = torch.cat(self.epoch_statistics['frame_id'], dim=0)    # N,
        camera_id = torch.arange(quat_values.shape[1])    # P,
        rend_mask_iou_mp = torch.cat(self.epoch_statistics['rend_mask_iou_mp'], dim=0)    # N,P
        shape = torch.cat(self.epoch_statistics['shape'], dim=0)       # N,P,4
        texture_uv = torch.cat(self.epoch_statistics['texture_uv'], dim=0)       # N,P,4


        # Quats
        quat_scores_max, _ = quat_scores.max(dim=-1, keepdim=True)
        quat_scores_max = quat_scores_max.expand_as(quat_scores)
        quat_ismax = (quat_scores_max <= (quat_scores + 1e-6))         # N,P
        quat_ismax_flat = quat_ismax.view(-1)                                    # NP
        quat_max_idx = quat_ismax_flat.nonzero().squeeze(1)
        quat_notmax_idx = (~quat_ismax_flat).nonzero().squeeze(1)

        quat_values_all = quat_values.view(-1,4)                            # NP,4
        quat_values_max = quat_values_all.index_select(0, quat_max_idx)
        quat_values_notmax = quat_values_all.index_select(0, quat_notmax_idx)

        quat_error_sum = (quat_error * quat_scores).sum(-1)                 # N
        quat_error_mean = quat_error_sum.mean()                             # scalar
        quat_error_max = quat_error.view(-1).index_select(0, quat_max_idx)  # N
        quat_error_notmax = quat_error.view(-1).index_select(0, quat_notmax_idx)  # N

        rend_mask_iou_max = rend_mask_iou_mp.view(-1).index_select(0, quat_max_idx)  # N

        # UV
        network_quat_values_uv = geom_utils.camera_quat_to_position_az_el(network_quat_values, self.opts.initial_quat_bias_deg).detach().cpu().numpy()
        quat_values_max_uv = geom_utils.camera_quat_to_position_az_el(quat_values_max, self.opts.initial_quat_bias_deg).detach().cpu().numpy()
        quat_values_notmax_uv = geom_utils.camera_quat_to_position_az_el(quat_values_notmax, self.opts.initial_quat_bias_deg).detach().cpu().numpy()
        gt_quat_uv = geom_utils.camera_quat_to_position_az_el(gt_quat, self.opts.initial_quat_bias_deg).detach().cpu().numpy()


        # Max camera hypothesis distribution
        max_cam_ids = quat_ismax.nonzero() # Z,2
        max_cam_ids = max_cam_ids[:,1]  # camera ids for all max elements
        max_cam_ids_hist = quat_ismax.sum(dim=0).float()/quat_ismax.sum()
        fig = plt.figure()
        plt.bar(np.arange(len(max_cam_ids_hist)), max_cam_ids_hist)
        plt.title('Max camera hypothesis distribution')
        max_cam_ids_hist_img = tensorboardX.utils.figure_to_image(fig)

        # Histograms
        hist_kwargs = {'bins':100, 'range':[[-1.0001,1.0001]]*2, 'density':False}
        quat_max_hist,hxe,hye = visutil.hist2d(quat_values_max_uv, **hist_kwargs)
        quat_notmax_hist,_,_ = visutil.hist2d(quat_values_notmax_uv, **hist_kwargs)
        network_quat_hist,_,_ = visutil.hist2d(network_quat_values_uv, **hist_kwargs)
        quat_gt_hist,_,_ = visutil.hist2d(gt_quat_uv, **hist_kwargs)
        error_max_hist,_,_ = visutil.hist2d(quat_values_max_uv, weights=quat_error_max, **hist_kwargs)
        error_notmax_hist,_,_ = visutil.hist2d(quat_values_notmax_uv, weights=quat_error_notmax, **hist_kwargs)
        error_net_hist,_,_ = visutil.hist2d(network_quat_values_uv, weights=network_quat_error, **hist_kwargs)
        error_max_per_quat_hist = np.where(quat_max_hist>0,error_max_hist/quat_max_hist,0)
        error_notmax_per_quat_hist = np.where(quat_notmax_hist>0,error_notmax_hist/quat_notmax_hist,0)
        error_net_per_quat_hist = np.where(network_quat_hist>0,error_net_hist/network_quat_hist,0)

        # Blurred Histograms
        sigma = 0.75
        quat_max_hist_blur = gaussian_filter(quat_max_hist, sigma)
        network_quat_hist_blur = gaussian_filter(network_quat_hist, sigma)
        quat_notmax_hist_blur = gaussian_filter(quat_notmax_hist, sigma)
        quat_gt_hist_blur = gaussian_filter(quat_gt_hist, sigma)
        error_max_hist_blur = gaussian_filter(error_max_hist, sigma)
        error_net_hist_blur = gaussian_filter(error_net_hist, sigma)
        error_notmax_hist_blur = gaussian_filter(error_notmax_hist, sigma)
        error_max_per_quat_hist_blur = gaussian_filter(error_max_per_quat_hist, sigma)
        error_notmax_per_quat_hist_blur = gaussian_filter(error_notmax_per_quat_hist, sigma)
        error_net_per_quat_hist_blur = gaussian_filter(error_net_per_quat_hist, sigma)

        sss = {
            'scalar':{
                'quat_error': float(quat_error_mean),
                'max_quat_error': float(quat_error_max.mean()),
                'net_quat_error': float(network_quat_error.mean()),
            },
            'hist': {
                'quat_error_sum': quat_error_sum,
                'quat_error_max': quat_error_max,
                'quat_error_all': quat_error.view(-1),
                'net_quat_error_all': network_quat_error.view(-1),
                'max_cam_ids': {
                    'values':max_cam_ids,
                    'bins':torch.arange(quat_scores.shape[1]+1),
                    'max_bins':quat_scores.shape[1],
                }
            },
            'img' : {
                'uv_hist': image_utils.concatenate_images2d([
                    [
                        visutil.hist2d_to_img(quat_max_hist,hxe,hye,title='quat_values_max_uv'),
                        visutil.hist2d_to_img(quat_notmax_hist,hxe,hye,title='quat_values_notmax_uv'),
                        visutil.hist2d_to_img(network_quat_hist,hxe,hye,title='network_quat_values_uv'),
                        visutil.hist2d_to_img(quat_gt_hist,hxe,hye,title='gt_quat_uv'),
                    ],
                    [
                        visutil.hist2d_to_img(error_max_per_quat_hist,hxe,hye,title='error_max_per_quat_hist'),
                        visutil.hist2d_to_img(error_notmax_per_quat_hist,hxe,hye,title='error_notmax_per_quat_hist'),
                        visutil.hist2d_to_img(error_net_per_quat_hist,hxe,hye,title='error_network_per_quat_hist'),
                    ],
                    [
                        visutil.hist2d_to_img(error_max_hist,hxe,hye,title='error_max_hist'),
                        visutil.hist2d_to_img(error_notmax_hist,hxe,hye,title='error_notmax_hist'),
                        visutil.hist2d_to_img(error_net_hist,hxe,hye,title='error_network_hist'),
                    ],
                ]),
                'uv_hist_blur': image_utils.concatenate_images2d([
                    [
                        visutil.hist2d_to_img(quat_max_hist_blur,hxe,hye,title='quat_values_max_uv'),
                        visutil.hist2d_to_img(quat_notmax_hist_blur,hxe,hye,title='quat_values_notmax_uv'),
                        visutil.hist2d_to_img(network_quat_hist_blur,hxe,hye,title='network_quat_values_uv'),
                        visutil.hist2d_to_img(quat_gt_hist_blur,hxe,hye,title='gt_quat_uv'),
                    ],
                    [
                        visutil.hist2d_to_img(error_max_per_quat_hist_blur,hxe,hye,title='error_max_per_quat_hist'),
                        visutil.hist2d_to_img(error_notmax_per_quat_hist_blur,hxe,hye,title='error_notmax_per_quat_hist'),
                        visutil.hist2d_to_img(error_net_per_quat_hist_blur,hxe,hye,title='error_network_per_quat_hist'),
                    ],
                    [
                        visutil.hist2d_to_img(error_max_hist_blur,hxe,hye,title='error_max_hist'),
                        visutil.hist2d_to_img(error_notmax_hist_blur,hxe,hye,title='error_notmax_hist'),
                        visutil.hist2d_to_img(error_net_hist_blur,hxe,hye,title='error_network_hist'),
                    ],
                ]),
                'max_cam_ids_hist': max_cam_ids_hist_img,
            },
            'raw': {
                'gt_quat':gt_quat.numpy(),
                'gt_cam':gt_cam.numpy(),
                'network_quat_values':network_quat_values.numpy(),
                'network_cam_values':network_cam_values.numpy(),
                'quat_values':quat_values.numpy(),
                'cam_values':cam_values.numpy(),
                'quat_scores':quat_scores.numpy(),
                'quat_error':quat_error.numpy(),
                'network_quat_error':network_quat_error.numpy(),
                'frame_id':frame_id.numpy(),
                'camera_id':camera_id.numpy(),
                'rend_mask_iou_max':rend_mask_iou_max.numpy(),
                'rend_mask_iou_mp':rend_mask_iou_mp.numpy(),
                'shape':shape.numpy(),
                'texture_uv':texture_uv.numpy(),
                'hist':{
                    'hxe':hxe,
                    'hye':hye,
                    'quat_max_hist': quat_max_hist,
                    'quat_notmax_hist': quat_notmax_hist,
                    'quat_gt_hist': quat_gt_hist,
                    'error_max_hist': error_max_hist,
                    'error_notmax_hist': error_notmax_hist,
                    'error_max_per_quat_hist': error_max_per_quat_hist,
                    'error_notmax_per_quat_hist': error_notmax_per_quat_hist
                }
            }
        }

        return sss

    def get_param_groups(self):
        opts = self.opts
        param_groups = [
            {'params':self.model.module.model.mean_shape,},
        ]
        if opts.pred_pose:
            param_groups.append({'params':self.model.module.model.posePred.parameters(),})
        if opts.pred_shape:
            param_groups.append({'params':self.model.module.model.shapePred.parameters(),})
        if opts.pred_texture:
            param_groups.append({'params':self.model.module.model.texturePred.parameters(),})
        return param_groups

    def get_camera_param_groups(self):
        param_groups = [
            {
                'params':self.model.module.dataset_camera_params,
                'lr':self.opts.optimizeLR,
                'momentum':self.opts.optimizeMomentum,
            },
        ]
        return param_groups

def main(_):
    opts = flags.FLAGS
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)

    if opts.use_gt_camera:
        opts.num_multipose_az = 1
        opts.num_multipose_el = 1
    opts.num_multipose = opts.num_multipose_az * opts.num_multipose_el
    # torch.autograd.set_detect_anomaly(opts.debug)
    trainer = ShapeTrainer(opts)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
