from __future__ import absolute_import, division, print_function

import os

import matplotlib as mpl
from absl import app, flags

if 'DISPLAY' not in os.environ:
    print('Display not found. Using Agg backend for matplotlib')
    mpl.use('Agg')
else:
    print(f'Found display : {os.environ["DISPLAY"]}')

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

from collections import OrderedDict

import torchvision
from scipy.ndimage.filters import gaussian_filter

from ..data import cub as cub_data
from ..data import imagenet as imagenet_data
from ..data import json_dataset as json_data
from ..data import p3d as p3d_data
from ..nnutils import geom_utils, loss_utils, train_utils
from ..nnutils.architecture import ShapeCamTexNet
from ..utils import bird_vis
from ..utils import image as image_utils
from ..utils import mesh
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
flags.DEFINE_boolean('laplacianDeltaV', False, 'Smooth DeltaV instead of V in laplacian_loss')


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
flags.DEFINE_boolean('optimizeCameraCont', False, 'Optimize Camera Continuously')
flags.DEFINE_float('optimizeLR', 0.0001, 'Learning rate for camera pose optimization')
flags.DEFINE_float('optimizeMomentum', 0, 'Momentum for camera pose optimization')
flags.DEFINE_enum('optimizeAlgo', 'adam', ['sgd','adam'], 'Algo for camera pose optimization')
flags.DEFINE_float('quatScorePeakiness', 20, 'quat score = e^(-peakiness * loss)')
flags.DEFINE_string('cameraPoseDict', '', 'Path to pre-computed camera pose dict for entire dataset')


flags.DEFINE_float('softras_sigma', 1e-5, 'Softras sigma (transparency)')
flags.DEFINE_float('softras_gamma', 1e-4, 'Softras gamma (blurriness)')

flags.DEFINE_boolean('texture_flipCam', False, 'Render flipped mesh and supervise using flipped image')

flags.DEFINE_boolean('optimizeCamera_reloadCamsFromDict', False, 'Ignore camera pose from saved dictionary')

flags.DEFINE_float('optimizeAzRange', 30, 'Optimize Azimuth range (degrees')
flags.DEFINE_float('optimizeElRange', 30, 'Optimize Elevation range (degrees')
flags.DEFINE_float('optimizeCrRange', 60, 'Optimize CycloRotation range (degrees')

class PoseTrainer(train_utils.Trainer):
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

        ####################
        # Define model
        ####################
        img_size = (opts.img_size, opts.img_size)
        self.model = ShapeCamTexNet(
            img_size, opts.nz_feat, opts.perspective, opts.symmetric_mesh,
            pred_shape=opts.pred_shape, pred_pose=opts.pred_pose, pred_texture=opts.pred_texture,
            mean_shape=self.verts, faces=self.faces,
            verts_uv_vx2=self.verts_uv, faces_uv_fx3x2=self.faces_uv,
            texture_opts=opts
        )

        # Load encoder, shape, texture and pose networks!
        if opts.pretrained_network_path:
            self.load_network(self.model, 'pred', 0, path=opts.pretrained_network_path)

        # Disable gradients for encoder, shape and texture
        for subnet in [self.model.convEncoder, self.model.shapePred, self.model.texturePred]:
            if subnet is not None:
                subnet.eval()
                for param in subnet.parameters():
                    param.requires_grad = False
        self.model.mean_shape.requires_grad = False

        if not opts.is_train:
            self.model.eval()

        self.model = self.model.cuda()

        # Store UV sperical texture map
        verts_sph = geom_utils.convert_uv_to_3d_coordinates(verts_uv)
        if not opts.textureUnwrapUV:
                uv_sampler = mesh.compute_uvsampler_softras(verts_sph, faces, tex_size=opts.tex_size, shift_uv=False)
        else:
                uv_sampler = mesh.compute_uvsampler_softras_unwrapUV(faces_uv, faces, tex_size=opts.tex_size, shift_uv=False)

        uv_texture = visutil.uv2bgr(uv_sampler) # F,T,T,3
        uv_texture = np.repeat(uv_texture[:,:,:,None,:], opts.tex_size, axis=3) # F,T,T,T,2
        self.uv_texture = torch.tensor(uv_texture).float().cuda()/255.

        self.vis_rend = bird_vis.VisRenderer(opts.img_size, faces, perspective=opts.perspective)

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

    def define_criterion(self):
        pass

    def set_input(self, batch):
        opts = self.opts

        # Batch:
        #   tracklet_id: N
        #   frame_id: N, 2
        #   img:  N, 2, ch, h, w
        #   mask: N, ch, h, w
        tracklet_id, frame_id, img, masks = None, batch['inds'], batch['img'], batch['mask']
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

        input_img = img.clone()
        for b in range(input_img.size(0)):
            input_img[b] = self.resnet_transform(input_img[b])

        img = img.float()
        input_img = input_img.float()
        img = img.float()
        masks = masks.float()

        self.tracklet_id = tracklet_id
        self.frame_id = frame_id
        self.img = img.cuda(non_blocking=True)
        self.input_img = input_img.cuda(non_blocking=True)
        self.mask = masks.cuda(non_blocking=True)
        self.gt_camera_pose = gt_camera_pose.cuda(non_blocking=True)

        if opts.flip_train:
            self.flip_frame_id = int(1e6) - self.frame_id
            self.flip_input_img = torch.flip(self.input_img, [-1])
            self.flip_img = torch.flip(self.img, [-1])
            self.flip_mask = torch.flip(self.mask, [-1])
            self.flip_mask_dt = torch.flip(self.mask_dt, [-1])
            self.flip_mask_dt_barrier = torch.flip(self.mask_dt_barrier, [-1])

            if hasattr(self,'gt_camera_pose'):
                self.flip_gt_camera_pose = geom_utils.reflect_cam_pose(self.gt_camera_pose)

    def forward(self, visualize_camera_dist=False, **kwargs):
        opts = self.opts

        delta_v, texture_flow, cams_bx7 = self.model(self.input_img)
        self.total_loss = loss_utils.camera_loss(cams_bx7, self.gt_camera_pose, 0)

        if opts.pred_texture:
            if opts.texture_predict_flow:
                textures = geom_utils.sample_textures(texture_flow, self.orig_img)
            else:
                textures = texture_flow

        if opts.pred_texture:
            uv_flows = self.model.texturePred.uvimage_pred.detach()
            if opts.texture_predict_flow:
                # B x 2 x H x W
                # B x H x W x 2
                self.uv_flows = uv_flows.permute(0, 2, 3, 1)
                self.uv_images = torch.nn.functional.grid_sample(self.img,
                                                                self.uv_flows)
            else:
                self.uv_images = uv_flows

        # delta_v = delta_v*opts.delta_v_scale
        mean_shape = self.model.get_mean_shape()
        pred_v = mean_shape[None,:,:] + (delta_v)

        self.shape = pred_v.detach()
        self.cams_bx7 = cams_bx7.detach()

        if visualize_camera_dist:
            self.return_dict = {
                'statistics': {
                    'frame_id':self.frame_id.detach().cpu(),
                    'gt_cam':self.gt_camera_pose.detach().cpu(),
                    'pred_cam':cams_bx7.detach().cpu(),
                    'shape':torch.tensor([0.]),
                    'texture_uv':torch.tensor([0.]),
                }
            }

            self.update_epoch_statistics()

    def get_current_visuals(self):
        # try:
        vis_dict = {'img':{}, 'mesh':{}, 'video':{}, 'text':{}}
        num_show = min(8,self.cams_bx7.shape[0])
        img = self.img.cpu().detach()
        mask = self.mask.cpu().detach()
        for i in range(num_show):
            rend_predcam = self.vis_rend(self.shape[i], self.cams_bx7[i], texture=self.uv_texture)
            rend_gtcam = self.vis_rend(self.shape[i], self.gt_camera_pose[i], texture=self.uv_texture)
            iii = bird_vis.tensor2im(img[i])
            mmm = bird_vis.tensor2mask(mask[i])
            best_imgs_list = [[iii,mmm,rend_predcam,rend_gtcam]]
            vis_dict['img']['%d/0_best' % i] = image_utils.concatenate_images2d(best_imgs_list)[:,:,::-1]
        return vis_dict
        # except:
        #     return {}

    def get_current_scalars(self):
        sc_dict = [
            ('total_loss', self.total_loss.item()),
        ]
        sc_dict = OrderedDict(sc_dict)
        return sc_dict

    def save(self, epoch_prefix):
        '''Saves the model.'''
        self.save_network(self.model, 'pred', epoch_prefix)
        self.save_network(self.model.posePred, 'posenet', epoch_prefix)
        return

    def reset_epoch_statistics(self):
        self.epoch_statistics = {
            'frame_id':[],      # N,4    gt quat
            'gt_cam':[],      # N,4    gt quat
            'pred_cam':[],  # N,P,4  net-predicted camera pose
            'shape':[],  # N,P,4  net-predicted camera pose
            'texture_uv':[],  # N,P,4  net-predicted camera pose
        }

    def update_epoch_statistics(self):
        statistics = self.return_dict['statistics']
        self.epoch_statistics['frame_id'].append(statistics['frame_id'].detach().cpu())
        self.epoch_statistics['gt_cam'].append(statistics['gt_cam'].detach().cpu())
        self.epoch_statistics['pred_cam'].append(statistics['pred_cam'].detach().cpu())
        self.epoch_statistics['shape'].append(statistics['shape'].detach().cpu())
        self.epoch_statistics['texture_uv'].append(statistics['texture_uv'].detach().cpu())

    def get_epoch_statistics(self):
        frame_id    = torch.cat(self.epoch_statistics['frame_id'], dim=0)       # N,
        gt_cam      = torch.cat(self.epoch_statistics['gt_cam'], dim=0)         # N,4
        pred_cam    = torch.cat(self.epoch_statistics['pred_cam'], dim=0)       # N,P,4
        shape    = torch.cat(self.epoch_statistics['shape'], dim=0)       # N,P,4
        texture_uv    = torch.cat(self.epoch_statistics['texture_uv'], dim=0)       # N,P,4

        gt_quat = gt_cam[...,3:7]
        network_quat_values = pred_cam[...,3:7]

        quat_error = geom_utils.hamilton_product(network_quat_values, geom_utils.quat_inverse(gt_quat))
        _, quat_error = geom_utils.quat2axisangle(quat_error)
        quat_error = torch.min(quat_error, 2*np.pi - quat_error)
        network_quat_error = quat_error * 180 / np.pi

        network_quat_values = geom_utils.quat_to_camera_position(network_quat_values, self.opts.initial_quat_bias_deg)
        gt_quat = geom_utils.quat_to_camera_position(gt_quat, self.opts.initial_quat_bias_deg)                          # N,3

        # UV
        network_quat_values_uv = geom_utils.convert_3d_to_uv_coordinates(network_quat_values).detach().cpu().numpy()
        gt_quat_uv = geom_utils.convert_3d_to_uv_coordinates(gt_quat).detach().cpu().numpy()

        # Histograms
        hist_kwargs = {'bins':100, 'range':[[-1.0001,1.0001]]*2, 'density':False}
        network_quat_hist,hxe,hye = visutil.hist2d(network_quat_values_uv, **hist_kwargs)
        quat_gt_hist,_,_ = visutil.hist2d(gt_quat_uv, **hist_kwargs)
        error_net_hist,_,_ = visutil.hist2d(network_quat_values_uv, weights=network_quat_error, **hist_kwargs)
        error_net_per_quat_hist = np.where(network_quat_hist>0,error_net_hist/network_quat_hist,0)

        # Blurred Histograms
        sigma = 0.75
        network_quat_hist_blur = gaussian_filter(network_quat_hist, sigma)
        quat_gt_hist_blur = gaussian_filter(quat_gt_hist, sigma)
        error_net_hist_blur = gaussian_filter(error_net_hist, sigma)
        error_net_per_quat_hist_blur = gaussian_filter(error_net_per_quat_hist, sigma)

        sss = {
            'scalar':{
                'net_quat_error': float(network_quat_error.mean()),
            },
            'hist': {
                'net_quat_error_all': network_quat_error.view(-1),
            },
            'img' : {
                'uv_hist': image_utils.concatenate_images2d([
                    [
                        visutil.hist2d_to_img(network_quat_hist,hxe,hye,title='network_quat_values_uv'),
                        visutil.hist2d_to_img(quat_gt_hist,hxe,hye,title='gt_quat_uv'),
                    ],
                    [
                        visutil.hist2d_to_img(error_net_per_quat_hist,hxe,hye,title='error_network_per_quat_hist'),
                        visutil.hist2d_to_img(error_net_hist,hxe,hye,title='error_network_hist'),
                    ]
                ]),
                'uv_hist_blur': image_utils.concatenate_images2d([
                    [
                        visutil.hist2d_to_img(network_quat_hist_blur,hxe,hye,title='network_quat_values_uv'),
                        visutil.hist2d_to_img(quat_gt_hist_blur,hxe,hye,title='gt_quat_uv'),
                    ],
                    [
                        visutil.hist2d_to_img(error_net_per_quat_hist_blur,hxe,hye,title='error_network_per_quat_hist'),
                        visutil.hist2d_to_img(error_net_hist_blur,hxe,hye,title='error_network_hist'),
                    ],
                ]),
            },
            'raw': {
                'gt_cam':gt_cam.numpy(),
                'cam_values':pred_cam[...,None,:].numpy(),
                'quat_scores':(torch.zeros_like(gt_cam[...,0:1])).numpy(),
                'frame_id':frame_id.numpy(),
                'shape':shape.numpy(),
                'texture_uv':texture_uv.numpy(),
            }
        }

        return sss

    def get_param_groups(self):
        opts = self.opts
        param_groups = [
            # Pass only pose network parameters
            {'params':self.model.posePred.parameters(), 'lr':opts.learning_rate},
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
    trainer = PoseTrainer(opts)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
