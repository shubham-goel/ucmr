from __future__ import absolute_import, division, print_function

import os

import matplotlib as mpl
from absl import app, flags

if 'DISPLAY' not in os.environ:
    print('Display not found. Using Agg backend for matplotlib')
    mpl.use('Agg')
else:
    print(f'Found display : {os.environ["DISPLAY"]}')
import os.path as osp
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorboardX
import torch
import torchvision
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

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
from ..utils import mesh, visutil


flags.DEFINE_string('dataset', 'cub', 'Dataset')

flags.DEFINE_boolean('perspective', False, 'whether to use strong perrspective projection')
flags.DEFINE_string('shape_path', 'birds/cmr_mean_birds_shape.npy', 'Path to initial mean shape')

flags.DEFINE_integer('num_multipose', -1, 'num_multipose_az * num_multipose_el')
flags.DEFINE_integer('num_multipose_az', 8, 'Number of camera pose hypothesis bins (along azimuth)')
flags.DEFINE_integer('num_multipose_el', 1, 'Number of camera pose hypothesis bins (along elevation)')

flags.DEFINE_float('initial_quat_bias_deg', 90, 'Rotation bias in deg. 90 for head-view, 45 for breast-view')
flags.DEFINE_enum('renderer', 'nmr', ['nmr','softras'], 'Which renderer to use')

flags.DEFINE_float('scale_bias', 0.8, 'Scale bias for camera pose')
flags.DEFINE_boolean('optimizeCameraCont', False, 'Optimize Camera continuously')
flags.DEFINE_boolean('optimizeScale', True, 'Optimize Scale')
flags.DEFINE_boolean('optimizeTrans', True, 'Optimize Trans')
flags.DEFINE_boolean('optimizeAz', True, 'Optimize Azimuth')
flags.DEFINE_boolean('optimizeEl', True, 'Optimize Elevation')
flags.DEFINE_boolean('optimizeCyRot', True, 'Optimize Cyclic rotation')
flags.DEFINE_integer('optimizeSteps', 150, 'Number of setps for camera pose optimization')
flags.DEFINE_float('optimizeLR', 0.1, 'Learning rate for camera pose optimization')
flags.DEFINE_float('optimizeBeta1', 0.9, 'Adam momentum for camera pose optimization')
flags.DEFINE_enum('optimizeAlgo', 'adam', ['sgd','adam'], 'Algo for camera pose optimization')
flags.DEFINE_float('quatScorePeakiness', 20, 'quat score = e^(-peakiness * loss)')
flags.DEFINE_string('cameraPoseDict', '', 'Path to pre-computed camera pose dict for entire dataset')
flags.DEFINE_boolean('quatScoreFixed', False, 'Is quat score fixed to last camera pose opt?')
flags.DEFINE_float('lossToScorePower', 1, 'transform loss = loss_rescaled ** power while computing score')

flags.DEFINE_boolean('tb_tag_frameid', False, 'Is tb tag = frameid?')

flags.DEFINE_float('optimizeAzRange', 30, 'Optimize Azimuth range (degrees')
flags.DEFINE_float('optimizeElRange', 30, 'Optimize Elevation range (degrees')
flags.DEFINE_float('optimizeCrRange', 60, 'Optimize CycloRotation range (degrees')

flags.DEFINE_float('baseQuat_elevationBias', 0, 'Increase elevation by this angle')
flags.DEFINE_float('baseQuat_azimuthBias', 0, 'Increase azimuth by this angle')

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '../../', 'cachedir')

class ForwardWrapper(torch.nn.Module):
    def __init__(self, opts, verts, faces, verts_uv):
        super().__init__()
        self.opts = opts

        self.register_buffer('verts_uv', verts_uv.float())
        self.register_buffer('verts', verts.float())
        self.register_buffer('faces', faces.long())

    def define_model(self):
        opts = self.opts

        img_size = (opts.img_size, opts.img_size)
        _fuv = self.verts_uv[self.faces]
        model = ShapeCamTexNet(
            img_size, opts.nz_feat, opts.perspective, opts.symmetric_mesh,
            opts.pred_shape, opts.pred_pose, opts.pred_texture,
            mean_shape=self.verts, faces=self.faces,
            verts_uv_vx2=self.verts_uv, faces_uv_fx3x2=_fuv,
            texture_opts=opts
        )
        self.register_buffer('model_mean_shape', model.get_mean_shape())

        # Renderers
        if opts.renderer == 'nmr':
            self.renderer_mask = NeuralRenderer(opts.img_size, perspective=opts.perspective)
        elif opts.renderer == 'softras':
            self.renderer_mask = SoftRas(opts.img_size, perspective=opts.perspective)
        self.vis_rend = bird_vis.VisRenderer(opts.img_size, self.faces, perspective=opts.perspective)

    def define_criterion(self):
        self.rend_mask_loss_fn = loss_utils.mask_l2_dt_loss

    def optimize_camera_pose(self, init_params, base_quat, pred_v, mask, mask_dt=None):
        """
        init_params : ..., scale trans az el cr
        base_quat : ...,4
        mask : ...,h,w
        pred_v : ...,N,3
        """
        opts = self.opts
        cam_params = init_params.detach().clone()
        base_quat = base_quat.detach()
        pred_v = pred_v.detach()
        mask = mask.detach()

        parameters = []

        scale = torch.nn.Parameter(cam_params[...,0:1])
        trans = torch.nn.Parameter(cam_params[...,1:3])
        az_param = torch.nn.Parameter(cam_params[...,3:4])
        el_param = torch.nn.Parameter(cam_params[...,4:5])
        cr_param = torch.nn.Parameter(cam_params[...,5:6])

        parameters.append(scale)
        parameters.append(trans)
        parameters.append(az_param)
        parameters.append(el_param)
        parameters.append(cr_param)

        if opts.optimizeAlgo=='adam':
            optimizer = torch.optim.Adam(parameters, lr=opts.optimizeLR, betas=(opts.optimizeBeta1, 0.999))
        elif opts.optimizeAlgo=='sgd':
            optimizer = torch.optim.SGD(parameters, lr=opts.optimizeLR)
        else:
            raise ValueError

        pbar = tqdm(range(opts.optimizeSteps),dynamic_ncols=True,desc=f'cam_opt')
        for _ in pbar:
            optimizer.zero_grad()

            az = torch.tanh(0.1 * az_param) * np.pi * opts.optimizeAzRange/180   # max 30 deg
            el = torch.tanh(0.1 * el_param) * np.pi * opts.optimizeElRange/180   # max 30 deg
            cr = torch.tanh(0.1 * cr_param) * np.pi * opts.optimizeCrRange/180   # max 60 deg

            cam_params = torch.cat([scale, trans, az, el, cr], dim=-1)
            quat = geom_utils.azElRot_to_quat(cam_params[...,3:6])
            quat = geom_utils.hamilton_product(quat, base_quat)
            cam_pred = torch.cat((cam_params[...,:3], quat), dim=-1) # BP,7

            faces_batch = self.faces[None,:,:].expand(pred_v.shape[0],-1,-1)
            rend_mask = self.renderer_mask.forward(pred_v, faces_batch.int(), cam_pred)
            loss = self.rend_mask_loss_fn(rend_mask, mask, reduction='none', mask2_dt=mask_dt).mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix(rloss=loss.item())

        cam_params = torch.cat([scale, trans, az_param, el_param, cr_param], dim=-1)
        return cam_params.detach()

    def forward(self, input_dict):
        """
        input dict = {
            img: normalized images
            mask: silhouette
            gt_camera_pose: (optional) GT camera pose
        }
        """
        opts = self.opts
        self.real_iter = input_dict['real_iter']

        self.input_img = input_dict['img']
        self.input_mask = input_dict['mask']
        self.input_mask_dt = input_dict['mask_dt']
        self.frame_id = input_dict['frame_id']
        self.cam_poses = input_dict['cam_poses']
        self.cam_scores = input_dict['cam_scores']
        self.gt_camera_pose = input_dict['gt_camera_pose']

        N, h, w = self.input_mask.shape
        assert(self.input_img.shape[0] == N)
        assert(self.input_img.shape[2:] == (h, w))
        self.input_N = N
        self.input_h = h
        self.input_w = w
        self.input_bs = N

        # Expand mask for multi_pose
        self.mask = self.input_mask[:,None,:,:].repeat(1,opts.num_multipose,1,1)
        self.mask = self.mask.view(self.input_bs*opts.num_multipose, h, w)
        self.mask_dt = self.input_mask_dt[:,None,:,:].repeat(1,opts.num_multipose,1,1)
        self.mask_dt = self.mask_dt.view(self.input_bs*opts.num_multipose, h, w)

        NUM_BATCH_POSE = self.input_bs*opts.num_multipose
        mean_shape = self.model_mean_shape
        pred_v = mean_shape[None,:,:].expand(NUM_BATCH_POSE, -1, -1)

        ############## Optimize Camera Pose
        ## COMPUTE CAMERA POSE
        # init parameters
        base_quat = self.cam_poses[...,3:7]                                 # N,P,4
        base_score = self.cam_scores                                        # N,P
        cam_params_init = torch.zeros((self.input_N,opts.num_multipose,1+2+3),
                                        dtype=base_quat.dtype,
                                        device=base_quat.device)        # N,P,6
        cam_params_init[...,0:3] = self.cam_poses[...,0:3]   # scale, trans, az_el_rot

        base_quat = base_quat.view((NUM_BATCH_POSE,) + base_quat.shape[2:]) # BP,4
        cam_params_init = cam_params_init.view((NUM_BATCH_POSE,) + cam_params_init.shape[2:]) # BP,6
        cam_params_final = self.optimize_camera_pose(cam_params_init,
                                                    base_quat,
                                                    pred_v,
                                                    self.mask,
                                                    self.mask_dt)  # BP,6
        # Camera parameters to quat
        az = torch.tanh(0.1 * cam_params_final[...,3]) * np.pi * opts.optimizeAzRange/180   # max 30 deg
        el = torch.tanh(0.1 * cam_params_final[...,4]) * np.pi * opts.optimizeElRange/180   # max 30 deg
        cr = torch.tanh(0.1 * cam_params_final[...,5]) * np.pi * opts.optimizeCrRange/180   # max 60 deg
        azelcr = torch.stack((az,el,cr), dim=-1)
        quat = geom_utils.azElRot_to_quat(azelcr)
        quat = geom_utils.hamilton_product(quat, base_quat)
        cam_pred = torch.cat((cam_params_final[...,:3], quat), dim=-1) # BP,7
        ######################

        # Render mask
        faces_batch = self.faces[None,:,:].expand(pred_v.shape[0],-1,-1)
        rend_mask = self.renderer_mask.forward(pred_v, faces_batch.int(), cam_pred)

        assert(torch.isfinite(mean_shape).all())
        assert(torch.isfinite(pred_v).all())
        assert(torch.isfinite(cam_pred).all())

        # Calculate quaternion error
        quat = cam_pred[:,3:7].view(self.input_bs, opts.num_multipose, 4)
        quat_error = geom_utils.hamilton_product(quat, geom_utils.quat_inverse(self.gt_camera_pose[:,None,3:7]))
        _, quat_error = geom_utils.quat2axisangle(quat_error)
        quat_error = torch.min(quat_error, 2*np.pi - quat_error)
        self.quat_error = quat_error * 180 / np.pi


        ### Loss computation ###
        ## Rend mask loss
        self.rend_mask_loss_mp = self.rend_mask_loss_fn(rend_mask, self.mask, reduction='none', mask2_dt=self.mask_dt) \
                                                .mean(dim=(-1,-2))                              \
                                                .view(self.input_bs,opts.num_multipose)
        self.total_loss_mp = self.rend_mask_loss_mp

        # Score from total_loss_mp
        loss_min,_ = self.total_loss_mp.min(dim=1)
        loss_max,_ = self.total_loss_mp.max(dim=1)
        loss_rescaled = (self.total_loss_mp - loss_min[:,None])/(loss_max[:,None] - loss_min[:,None])
        loss_rescaled = loss_rescaled.pow(opts.lossToScorePower)
        self.quat_score = torch.nn.functional.softmin(loss_rescaled * opts.quatScorePeakiness, dim=1)
        self.rend_mask_loss = (self.rend_mask_loss_mp*self.quat_score).sum(dim=1).mean()
        self.total_loss = self.rend_mask_loss

        ### Save variables in natural dimensions for visualization
        self.cam_pred = cam_pred.view(self.input_bs,opts.num_multipose,7)
        self.pred_v = pred_v.view((self.input_bs,opts.num_multipose,)+pred_v.shape[1:])[:,0,:,:]
        self.mean_shape = mean_shape

        ### Statistics
        self.statistics = {}
        self.statistics['quat_values'] = self.cam_pred[...,3:7] # N,P,4
        self.statistics['quat_scores'] = self.quat_score        # N,P
        self.statistics['gt_quat'] = self.gt_camera_pose[:,3:7] # N,4
        self.statistics['quat_error'] = self.quat_error         # N,P

        self.total_loss_num = torch.zeros_like(self.total_loss) + len(self.input_img)
        return_attr = {
            ## Inputs
            'input_img',
            'input_mask',
            'frame_id',
            'gt_camera_pose',

            ## Losses
            'total_loss_num',
            'total_loss',
            'total_loss_mp',
            'rend_mask_loss',

            ## Intermediate values
            'quat_score',
            'pred_v',
            'cam_pred',
            'mean_shape',
            'quat_error',

            # Statistics
            'statistics',
        }
        return_dict = {
            k:getattr(self,k) for k in return_attr
        }
        return return_dict

class MultiplexOptimizer(train_utils.Trainer):
    def define_model(self):
        opts = self.opts

        mean_shape = mesh.fetch_mean_shape(opts.shape_path, mean_centre_vertices=True)
        verts = mean_shape['verts']
        faces = mean_shape['faces']
        verts_uv = mean_shape['verts_uv']
        faces_uv = mean_shape['faces_uv']

        self.verts_uv = torch.from_numpy(verts_uv).float() # V,3
        self.verts = torch.from_numpy(verts).float() # V,3
        self.faces = torch.from_numpy(faces).long()  # F,2

        # Store UV sperical texture map
        verts_sph = geom_utils.convert_uv_to_3d_coordinates(verts_uv)
        uv_sampler = mesh.compute_uvsampler(verts_sph, faces, tex_size=opts.tex_size)
        uv_texture = visutil.uv2bgr(uv_sampler) # F,T,T,3
        uv_texture = np.repeat(uv_texture[:,:,:,None,:], opts.tex_size, axis=3) # F,T,T,T,2
        self.uv_texture = torch.tensor(uv_texture).float().cuda()/255.

        # Renderer for visualization
        self.vis_rend = bird_vis.VisRenderer(opts.img_size, faces, perspective=opts.perspective)

        ####################
        # Define model
        ####################
        self.model = ForwardWrapper(opts, self.verts, self.faces, self.verts_uv)
        self.model.define_model()
        self.model.define_criterion()
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

        ## Dictionary of all camera poses
        # dict[frameid] = (Px7: [scale trans quat], P:score, 3:gtscale gttrans)
        # Store camera pose of un-flipped image only
        if opts.cameraPoseDict:
            self.datasetCameraPoseDict = np.load(opts.cameraPoseDict, allow_pickle=True)['campose'].item()
            print(f'Loaded cam_pose_dict of size {len(self.datasetCameraPoseDict)} (should be 5964)')
        else:
            self.datasetCameraPoseDict = {}
        self.base_quats = geom_utils.get_base_quaternions(num_pose_az=opts.num_multipose_az,
                                                        num_pose_el=opts.num_multipose_el,
                                                        initial_quat_bias_deg=opts.initial_quat_bias_deg,
                                                        elevation_bias=opts.baseQuat_elevationBias,
                                                        azimuth_bias=opts.baseQuat_azimuthBias)

        return

    def get_camera_multiplex(self, frame_id, gt_st0):

        frame_id_orig = min(frame_id, int(1e6)-frame_id)
        if frame_id != frame_id_orig:
            gt_st0 = gt_st0.clone()
            gt_st0[1] *= -1


        try:
            cam_pose, score, gt_st1 = self.datasetCameraPoseDict[frame_id_orig]
            cam_pose = cam_pose.clone()
            return None, None
        except KeyError:
            quats = self.base_quats.clone()
            trans = torch.zeros(quats.shape[0],2).float()
            scale = torch.zeros(quats.shape[0],1).float() + self.opts.scale_bias
            cam_pose = torch.cat([scale,trans,quats], dim=-1)
            score = torch.ones(cam_pose.shape[0]).float()/cam_pose.shape[0]
            gt_st1 = gt_st0

        cam_pose[...,0] *= gt_st0[0]/gt_st1[0]
        cam_pose[...,1:3] += gt_st0[1:3] - gt_st1[1:3]

        if frame_id == frame_id_orig:
            return cam_pose, score
        else:
            return self.reflect_cam_pose(cam_pose), score

    def update_camera_multiplex(self, frame_id, gt_st0, cam_pose, score):
        frame_id_orig = min(frame_id, int(1e6)-frame_id)
        if frame_id != frame_id_orig:
            gt_st0 = gt_st0.clone()
            gt_st0[1] *= -1
            cam_pose = self.reflect_cam_pose(cam_pose)
        self.datasetCameraPoseDict[frame_id_orig] = (cam_pose.detach().cpu(),
                                                score.detach().cpu(),
                                                gt_st0.detach().cpu())

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
            std=image_utils.BGR_STD)

    def define_criterion(self):
        pass

    def reflect_cam_pose(self, cam_pose):
        cam_pose = cam_pose * torch.tensor([1, -1, 1, 1, 1, -1, -1],
                                            dtype=cam_pose.dtype,
                                            device=cam_pose.device).view(1,-1)
        return cam_pose

    def set_input(self, batch):
        opts = self.opts

        # Batch:
        #   frame_id: N, 2
        #   img:  N, 2, ch, h, w
        #   mask: N, ch, h, w
        frame_id, img, masks, masks_dt = batch['inds'], batch['img'], batch['mask'], batch['mask_dt']
        masks = masks.unsqueeze(1) # Adding channel dimension
        masks = masks.unsqueeze(1) # Adding bb
        img = img.unsqueeze(1) # Adding bb

        # GT camera pose (from sfm_pose) for debugging
        gt_camera_pose = batch['sfm_pose'].float()

        N, bb, _, h, w = masks.shape
        assert(img.shape[:2] == (N, bb))
        assert(img.shape[3:] == (h, w))
        self.input_N = N
        self.input_bb = bb
        self.input_h = h
        self.input_w = w
        self.input_bs = N*bb

        img = img.view(N*bb, 3, h, w).float()
        masks = masks.view(N*bb, h, w).float()
        masks_dt = masks_dt.view(N*bb, h, w).float()

        input_img = img.clone()
        for b in range(input_img.size(0)):
            input_img[b] = self.resnet_transform(input_img[b])

        img = img.float()
        input_img = input_img.float()
        img = img.float()
        masks = masks.float()
        masks_dt = masks_dt.float()

        camera_poses_scores = [self.get_camera_multiplex(f.item(), gtcam[:3]) for f,gtcam in zip(frame_id,gt_camera_pose)]
        cam_poses = [c for c,s in camera_poses_scores]
        cam_scores = [s for c,s in camera_poses_scores]

        # Find inputs we'll be processing.
        # If opts.screwed_drop_last_batch, we only process inputs that
        # aren't already present in self.camera_pose_dict
        idx_ok = [i for i,c in enumerate(cam_poses) if c is not None]
        if len(idx_ok) == 0:
            self.invalid_batch = True
            return
        else:
            self.invalid_batch = False
        idx_ok = torch.tensor(idx_ok, dtype=torch.long, device=frame_id.device)
        minimum_required_batch_size = len(self.model.device_ids)
        while len(idx_ok) < minimum_required_batch_size:
            idx_ok = torch.cat([idx_ok, idx_ok[-1]], dim=0)

        cam_poses = torch.stack([cam_poses[i] for i in idx_ok], dim=0)
        cam_scores = torch.stack([cam_scores[i] for i in idx_ok], dim=0)
        frame_id = torch.index_select(frame_id, 0, idx_ok)
        img = torch.index_select(img, 0, idx_ok)
        input_img = torch.index_select(input_img, 0, idx_ok)
        masks = torch.index_select(masks, 0, idx_ok)
        masks_dt = torch.index_select(masks_dt, 0, idx_ok)
        gt_camera_pose = torch.index_select(gt_camera_pose, 0, idx_ok)

        self.frame_id = frame_id
        self.img = img.cuda(non_blocking=True)
        self.input_img = input_img.cuda(non_blocking=True)
        self.mask = masks.cuda(non_blocking=True)
        self.mask_dt = masks_dt.cuda(non_blocking=True)
        self.cam_poses = cam_poses.cuda(non_blocking=True)
        self.cam_scores = cam_scores.cuda(non_blocking=True)
        self.gt_camera_pose = gt_camera_pose.cuda(non_blocking=True)

    def forward(self, **kwargs):
        opts = self.opts

        input_dict = {
            'img':self.input_img,
            'mask':self.mask,
            'mask_dt':self.mask_dt,
            'real_iter':self.real_iter,
            'frame_id':self.frame_id,
            'cam_poses':self.cam_poses,
            'cam_scores':self.cam_scores,
        }
        if hasattr(self,'gt_camera_pose'):
            input_dict['gt_camera_pose'] = self.gt_camera_pose

        self.return_dict = self.model(input_dict)
        loss_keys = {
            # Losses
            'total_loss',
            'rend_mask_loss',
        }
        gpu_weights = self.return_dict['total_loss_num']/self.return_dict['total_loss_num'].sum()
        for k in loss_keys:
            setattr(self,k,(self.return_dict[k] * gpu_weights).mean())
        self.update_epoch_statistics()

        cam_pred = self.return_dict['cam_pred']
        gt_st = self.return_dict['gt_camera_pose'][...,0:3] # scale trans
        frame_id = self.return_dict['frame_id']
        quat_score = self.return_dict['quat_score']

        for i in range(frame_id.shape[0]):
            self.update_camera_multiplex(frame_id[i].item(), gt_st[i], cam_pred[i], quat_score[i])

    def get_current_visuals(self):
        vis_dict = {'img':{}, 'mesh':{}, 'video':{}, 'text':{}}

        opts = self.opts
        N = self.input_N
        bb = self.input_bb
        h = self.input_h
        w = self.input_w

        quat_score = self.return_dict['quat_score']
        pred_v = self.return_dict['pred_v']
        cam_pred = self.return_dict['cam_pred']
        total_loss_mp = self.return_dict['total_loss_mp'].cpu()
        quat_error = self.return_dict['quat_error']
        mean_shape = self.return_dict['mean_shape']
        if len(mean_shape.shape) > 2:
            mean_shape = mean_shape[0]

        images = self.return_dict['input_img']
        img_mask = self.return_dict['input_mask']
        frame_id = self.return_dict['frame_id']

        images = image_utils.unnormalize_img(images.detach().cpu())
        img_mask = img_mask.detach().cpu()
        frame_id = frame_id.detach().cpu()

        quat_score = quat_score.detach().cpu()
        pred_v = pred_v.detach()
        cam_pred = cam_pred.detach()

        num_show = min(32, images.shape[0])
        for i in range(num_show):
            img = bird_vis.tensor2im(images[i])
            mask = bird_vis.tensor2mask(img_mask[i])
            plot_imgs = [[img,mask]]

            viewpoints_imgs = []
            _, min_j = total_loss_mp[i,:].min(dim=0)
            for j in range(opts.num_multipose):
                rend_predcam = self.vis_rend(pred_v[i], cam_pred[i,j], texture=self.uv_texture)

                text_color = (0,255,0) if (min_j==j) else (255,255,255)
                text_img = rend_predcam
                textc = f'c{j} {quat_score[i,j]*100:.01f}%'
                text0 = f'L {total_loss_mp[i,j]:.03f}'
                text1 = f'E {quat_error[i,j]:.01f}d'
                text_img = cv2.putText(text_img,textc,(5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,text_color,1,cv2.LINE_AA)
                text_img = cv2.putText(text_img,text0,(5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,text_color,1,cv2.LINE_AA)
                text_img = cv2.putText(text_img,text1,(5,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,text_color,1,cv2.LINE_AA)
                viewpoints_imgs.append(text_img)

            for el in range(opts.num_multipose_el):
                row = []
                for az in range(opts.num_multipose_az):
                    iii = az*opts.num_multipose_el + el
                    row.append(viewpoints_imgs[iii])
                plot_imgs.append(row)

            tag_prefix = frame_id[i].item() if opts.tb_tag_frameid else i

            vis_dict['img']['%d/0' % tag_prefix] = image_utils.concatenate_images2d(plot_imgs)[:,:,::-1]
            vis_dict['img']['%d/0_best' % tag_prefix] = image_utils.concatenate_images1d([img,mask,viewpoints_imgs[min_j]])[:,:,::-1]
            vis_dict['text']['%d/0' % tag_prefix] = f'{frame_id[i].item()}'

        return vis_dict

    def get_current_scalars(self):
        loss_values = [
            'rend_mask_loss',
        ]
        sc_dict = [
            ('smoothed_total_loss', self.smoothed_total_loss),
            ('total_loss', self.total_loss.item()),
        ]
        for attr in loss_values:
            sc_dict.append((attr,getattr(self, attr).item()))
        sc_dict = OrderedDict(sc_dict)
        return sc_dict

    def save(self, epoch_prefix):
        '''Saves the model.'''
        return

    def reset_epoch_statistics(self):
        self.epoch_statistics = {
            'gt_quat':[],      # N,4    gt quat
            'quat_values':[],  # N,P,4  predicted camera pose
            'quat_scores':[],  # N,P    camera pose scores
            'quat_error':[],   # N,P    quat error in degrees
            'input_img':[],   # N,3,h,w input images
            'frame_id':[],     # N,     frame ids
        }

    def update_epoch_statistics(self):
        statistics = self.return_dict['statistics']
        self.epoch_statistics['gt_quat'].append(statistics['gt_quat'].detach().cpu())
        self.epoch_statistics['quat_values'].append(statistics['quat_values'].detach().cpu())
        self.epoch_statistics['quat_scores'].append(statistics['quat_scores'].detach().cpu())
        self.epoch_statistics['quat_error'].append(statistics['quat_error'].detach().cpu())
        self.epoch_statistics['input_img'].append(self.return_dict['input_img'].detach().cpu())
        self.epoch_statistics['frame_id'].append(self.return_dict['frame_id'].detach().cpu())

    def get_epoch_statistics(self):
        gt_quat =     torch.cat(self.epoch_statistics['gt_quat'], dim=0)     # N,4
        quat_values = torch.cat(self.epoch_statistics['quat_values'], dim=0) # N,P,4
        quat_scores = torch.cat(self.epoch_statistics['quat_scores'], dim=0) # N,P
        quat_error =  torch.cat(self.epoch_statistics['quat_error'], dim=0)  # N,P
        frame_id =    torch.cat(self.epoch_statistics['frame_id'], dim=0)    # N,
        camera_id =   torch.arange(quat_values.shape[1])    # P,


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

        quat_values_max = geom_utils.quat_to_camera_position(quat_values_max, self.opts.initial_quat_bias_deg)          # N,3
        quat_values_notmax = geom_utils.quat_to_camera_position(quat_values_notmax, self.opts.initial_quat_bias_deg)    # N(P-1),3
        gt_quat = geom_utils.quat_to_camera_position(gt_quat, self.opts.initial_quat_bias_deg)                          # N,3

        # Max camera hypothesis distribution
        max_cam_ids = quat_ismax.nonzero() # Z,2
        max_cam_ids = max_cam_ids[:,1]  # camera ids for all max elements
        max_cam_ids_hist = quat_ismax.sum(dim=0).float()/quat_ismax.sum()
        fig = plt.figure()
        plt.bar(np.arange(len(max_cam_ids_hist)), max_cam_ids_hist)
        plt.title('Max camera hypothesis distribution')
        max_cam_ids_hist_img = tensorboardX.utils.figure_to_image(fig)

        # UV
        quat_values_max_uv = geom_utils.convert_3d_to_uv_coordinates(quat_values_max).detach().cpu().numpy()
        quat_values_notmax_uv = geom_utils.convert_3d_to_uv_coordinates(quat_values_notmax).detach().cpu().numpy()
        gt_quat_uv = geom_utils.convert_3d_to_uv_coordinates(gt_quat).detach().cpu().numpy()

        # Image
        num_pose = quat_values.shape[1]
        all_quat_img = None

        # FrameID
        frame_id_gt = frame_id.squeeze(1)
        frame_id_all = frame_id_gt[:,None].expand_as(quat_scores).contiguous().view(-1)
        frame_id_max = frame_id_all.index_select(0, quat_max_idx)
        frame_id_notmax = frame_id_all.index_select(0, quat_notmax_idx)

        camera_id_gt = torch.zeros_like(frame_id_gt)-1
        camera_id_all = camera_id[None,:].expand_as(quat_scores).contiguous().view(-1)
        camera_id_max = camera_id_all.index_select(0, quat_max_idx)
        camera_id_notmax = camera_id_all.index_select(0, quat_notmax_idx)

        all_quat_mat = torch.cat([gt_quat,quat_values_max,quat_values_notmax,], dim=0)
        all_quat_frameid = torch.cat([frame_id_gt,frame_id_max,frame_id_notmax,], dim=0)
        all_quat_cameraid = torch.cat([camera_id_gt,camera_id_max,camera_id_notmax,], dim=0)
        all_quat_meta_cat = ['gt']*len(gt_quat)+['max']*len(quat_values_max)+['notmax']*len(quat_values_notmax)
        all_quat_meta = [(f'{cat}',f'{fid}',f'{cid}',f'{cat}:{cid}:{fid}') for cat,fid,cid in zip(all_quat_meta_cat, all_quat_frameid, all_quat_cameraid)]
        all_quat_meta_header = ('cat','frame_id','camera_id','cat:camid:fid')

        # Histograms
        hist_kwargs = {'bins':100, 'range':[[-1.0001,1.0001]]*2, 'density':False}
        quat_max_hist,hxe,hye = visutil.hist2d(quat_values_max_uv, **hist_kwargs)
        quat_notmax_hist,_,_ = visutil.hist2d(quat_values_notmax_uv, **hist_kwargs)
        quat_gt_hist,_,_ = visutil.hist2d(gt_quat_uv, **hist_kwargs)
        error_max_hist,_,_ = visutil.hist2d(quat_values_max_uv, weights=quat_error_max, **hist_kwargs)
        error_notmax_hist,_,_ = visutil.hist2d(quat_values_notmax_uv, weights=quat_error_notmax, **hist_kwargs)
        error_max_per_quat_hist = np.where(quat_max_hist>0,error_max_hist/quat_max_hist,0)
        error_notmax_per_quat_hist = np.where(quat_notmax_hist>0,error_notmax_hist/quat_notmax_hist,0)

        # Blurred Histograms
        sigma = 0.75
        quat_max_hist_blur = gaussian_filter(quat_max_hist, sigma)
        quat_notmax_hist_blur = gaussian_filter(quat_notmax_hist, sigma)
        quat_gt_hist_blur = gaussian_filter(quat_gt_hist, sigma)
        error_max_hist_blur = gaussian_filter(error_max_hist, sigma)
        error_notmax_hist_blur = gaussian_filter(error_notmax_hist, sigma)
        error_max_per_quat_hist_blur = gaussian_filter(error_max_per_quat_hist, sigma)
        error_notmax_per_quat_hist_blur = gaussian_filter(error_notmax_per_quat_hist, sigma)

        sss = {
            'embed':{
                'all_quat': {
                    'mat':all_quat_mat,
                    'metadata': all_quat_meta,
                    'label_img':all_quat_img,
                    'metadata_header':all_quat_meta_header,
                },
            },
            'scalar':{
                'quat_error': quat_error_mean,
            },
            'img' : {
                'uv_hist': image_utils.concatenate_images2d([
                    [
                        visutil.hist2d_to_img(quat_max_hist,hxe,hye,title='quat_values_max_uv'),
                        visutil.hist2d_to_img(quat_notmax_hist,hxe,hye,title='quat_values_notmax_uv'),
                        visutil.hist2d_to_img(quat_gt_hist,hxe,hye,title='gt_quat_uv'),
                    ],
                    [
                        visutil.hist2d_to_img(error_max_per_quat_hist,hxe,hye,title='error_max_per_quat_hist'),
                        visutil.hist2d_to_img(error_notmax_per_quat_hist,hxe,hye,title='error_notmax_per_quat_hist'),
                    ],
                    [
                        visutil.hist2d_to_img(error_max_hist,hxe,hye,title='error_max_hist'),
                        visutil.hist2d_to_img(error_notmax_hist,hxe,hye,title='error_notmax_hist'),
                    ],
                ]),
                'uv_hist_blur': image_utils.concatenate_images2d([
                    [
                        visutil.hist2d_to_img(quat_max_hist_blur,hxe,hye,title='quat_values_max_uv'),
                        visutil.hist2d_to_img(quat_notmax_hist_blur,hxe,hye,title='quat_values_notmax_uv'),
                        visutil.hist2d_to_img(quat_gt_hist_blur,hxe,hye,title='gt_quat_uv'),
                    ],
                    [
                        visutil.hist2d_to_img(error_max_per_quat_hist_blur,hxe,hye,title='error_max_per_quat_hist'),
                        visutil.hist2d_to_img(error_notmax_per_quat_hist_blur,hxe,hye,title='error_notmax_per_quat_hist'),
                    ],
                    [
                        visutil.hist2d_to_img(error_max_hist_blur,hxe,hye,title='error_max_hist'),
                        visutil.hist2d_to_img(error_notmax_hist_blur,hxe,hye,title='error_notmax_hist'),
                    ],
                ]),
                'max_cam_ids_hist': max_cam_ids_hist_img,
            },
            'raw': {
                'gt_quat':gt_quat.numpy(),
                'quat_values':quat_values.numpy(),
                'quat_scores':quat_scores.numpy(),
                'quat_error':quat_error.numpy(),
                'frame_id':frame_id.numpy(),
                'camera_id':camera_id.numpy(),
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
        return [torch.nn.Parameter(torch.tensor(0.))]

def main(_):
    opts = flags.FLAGS
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)

    opts.num_multipose = opts.num_multipose_az * opts.num_multipose_el
    # torch.autograd.set_detect_anomaly(opts.debug)
    trainer = MultiplexOptimizer(opts)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
