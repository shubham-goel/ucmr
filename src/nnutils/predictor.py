"""
Takes an image, returns stuff.
"""

from __future__ import absolute_import, division, print_function

import torch
import torchvision
from absl import app, flags

from ..nnutils import geom_utils
from ..nnutils.architecture import ShapeCamTexNet
from ..nnutils.nmr import SoftRas
from ..utils import bird_vis
from ..utils import image as image_utils
from ..utils import mesh

# These options are off by default, but used for some ablations reported.
flags.DEFINE_boolean('ignore_pred_delta_v', False, 'Use only mean shape for prediction')
flags.DEFINE_boolean('use_template_ms', False, 'Uses template mean shape for prediction')
flags.DEFINE_boolean('use_gt_camera', False, 'Uses sfm mean camera')

flags.DEFINE_string('shape_path', '', 'Path to initial mean shape')
flags.DEFINE_boolean('mean_centre_vertices', True, 'Mean-centre vertices (y,z only)')
flags.DEFINE_boolean('perspective', False, 'whether to use strong perrspective projection')

class MeshPredictor(object):
    def __init__(self, opts):
        self.opts = opts

        # Mean Shape
        mean_shape = mesh.fetch_mean_shape(opts.shape_path, mean_centre_vertices=opts.mean_centre_vertices)

        self.verts_uv = torch.from_numpy(mean_shape['verts_uv']).float().cuda() # V,2
        self.verts = torch.from_numpy(mean_shape['verts']).float().cuda() # V,3
        self.faces = torch.from_numpy(mean_shape['faces']).long().cuda()  # F,2
        self.faces_uv = torch.from_numpy(mean_shape['faces_uv']).float().cuda()  # F,3,2

        print('Setting up model..')
        img_size = (opts.img_size, opts.img_size)
        self.model = ShapeCamTexNet(
            img_size, opts.nz_feat, opts.perspective, opts.symmetric_mesh,
            pred_shape=opts.pred_shape, pred_pose=opts.pred_pose, pred_texture=opts.pred_texture,
            mean_shape=self.verts.cpu(), faces=self.faces.cpu(),
            verts_uv_vx2=self.verts_uv.cpu(), faces_uv_fx3x2=self.faces_uv.cpu(),
            texture_opts=opts
        )

        # Load model
        model_dict = torch.load(opts.pretrained_network_path, map_location='cpu')

        # Some stuff should be the same
        assert((self.model.faces == model_dict['faces']).all())
        assert((self.model.ilr_idx_inv == model_dict['ilr_idx_inv']).all())
        assert((self.model.left_sym_idx == model_dict['left_sym_idx']).all())
        assert((self.model.indep_idx == model_dict['indep_idx']).all())
        assert((self.model.right_partner_idx == model_dict['right_partner_idx']).all())
        assert((self.model.mean_shape_orig == model_dict['mean_shape_orig']).all())

        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.model = self.model.cuda()

        self.renderer = SoftRas(
                            opts.img_size,
                            perspective=opts.perspective,
                            light_intensity_ambient=1.0,
                            light_intensity_directionals=0.0
                        )
        self.renderer.ambient_light_only()
        self.renderer.renderer.set_gamma(1e-8)
        self.renderer.renderer.set_sigma(1e-8)
        self.renderer = self.renderer.cuda()

        self.vis_rend = bird_vis.VisRendererBatch(opts.img_size, self.faces.cpu().numpy(), perspective=opts.perspective)
        self.vis_rend.set_bgcolor([1., 1., 1.])
        self.vis_rend.set_light_dir([0, 1, -1], 0.38)

        # UV Sampler for resampling NMR texture
        verts_sph = geom_utils.convert_uv_to_3d_coordinates(self.verts_uv)
        self.uv_sampler = mesh.compute_uvsampler(
                                verts_sph.cpu().numpy(),
                                self.faces.cpu().numpy(),
                                tex_size=opts.tex_size,
                                shift_uv=opts.texture_uvshift
                            )
        self.uv_sampler = torch.tensor(self.uv_sampler).float().cuda()

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=image_utils.BGR_MEAN, std=image_utils.BGR_STD
        )


    def set_input(self, batch):
        opts = self.opts

        # original image where texture is sampled from.
        img_tensor = batch['img'].clone().type(torch.FloatTensor).cuda()

        # input_img is the input to resnet
        input_img_tensor = batch['img'].type(torch.FloatTensor).cuda()
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])

        self.input_imgs = input_img_tensor.cuda()
        self.imgs = img_tensor.cuda()
        if opts.use_gt_camera:
            cam_tensor = batch['sfm_pose'].type(torch.FloatTensor)
            self.sfm_cams = cam_tensor.cuda()

    def predict(self, batch):
        """
        batch has B x C x H x W numpy
        """
        self.set_input(batch)
        with torch.no_grad():
            self.forward()
        return self.collect_outputs()

    def forward(self):
        opts = self.opts
        N = self.input_imgs.shape[0]
        delta_v, texture_flow, cams_bx7 = self.model(self.input_imgs)

        if opts.use_gt_camera:
            self.cam_pred = self.sfm_cams
        else:
            self.cam_pred = cams_bx7

        # Deform mean shape:
        self.mean_shape = self.model.get_mean_shape()
        if opts.use_template_ms:
            self.pred_v = self.verts[None].expand(N,-1,-1)
        elif opts.ignore_pred_delta_v:
            self.pred_v = self.mean_shape + delta_v*0
        else:
            self.pred_v = self.mean_shape + delta_v

        # Texture stuff: texture_flow, textures
        if opts.pred_texture:
            if opts.texture_predict_flow:
                textures = geom_utils.sample_textures(texture_flow, self.orig_img)
            else:
                textures = texture_flow

            # print('textures', textures.contiguous().view(-1))
            tex_size = textures.size(2)
            textures = textures.unsqueeze(4).expand(-1, -1, -1, -1, tex_size, -1) # B,F,T,T,T,3
            self.textures = textures.detach()
            self.texture_flow = texture_flow.detach()
            self.texture_uvimage_pred = self.model.texturePred.uvimage_pred.detach()
        else:
            self.textures = None
            self.texture_flow = None
            self.texture_uvimage_pred = None

        # Render mask, texture
        self.rend_texture = None
        faces_batch = self.faces[None,:,:].expand(N,-1,-1)
        if opts.pred_texture:
            self.rend_texture, self.rend_mask = self.renderer.render_texture_mask(
                                                        self.pred_v,
                                                        faces_batch.int(),
                                                        self.cam_pred,
                                                        textures=textures
                                                    )
        else:
            self.rend_mask = self.renderer.forward(self.pred_v, faces_batch.int(), self.cam_pred)
            self.rend_texture = None


        if opts.pred_texture:
            uv_flows = self.model.texturePred.uvimage_pred.detach()
            if opts.texture_predict_flow:
                # B x 2 x H x W
                # B x H x W x 2
                self.uv_flows = uv_flows.permute(0, 2, 3, 1)
                self.uv_images = torch.nn.functional.grid_sample(self.imgs,
                                                                self.uv_flows)
            else:
                self.uv_images = uv_flows

    def collect_outputs(self):
        outputs = {
            # 'kp_pred': self.kp_pred.data,
            'verts': self.pred_v.detach(),
            # 'kp_verts': self.kp_verts.detach(),
            'cam_pred': self.cam_pred.detach(),
            'mask_pred': self.rend_mask.detach(),
        }
        if self.opts.texture and not self.opts.use_template_ms:
            outputs['texture'] = self.textures
            outputs['texture_pred'] = self.rend_texture.detach()
            outputs['uv_image'] = self.uv_images.detach()
            if self.opts.texture_predict_flow:
                outputs['uv_flow'] = self.uv_flows.detach()

        return outputs

    def resample_texture_nmr(self, uv_image):
        """ Resample texture for use with NMR
            uv_image: batch of texture uv-images
        """
        T = self.opts.tex_size
        F = self.faces.shape[0]
        B = uv_image.shape[0]
        uv_sampler_fxttx2 = self.uv_sampler.view(F, -1, 2)
        uv_sampler_bxfxttx2 = uv_sampler_fxttx2[None].expand(B, -1, -1, -1)
        tex_pred_bx3xfxtt = torch.nn.functional.grid_sample(uv_image, uv_sampler_bxfxttx2)
        tex_pred_bxfxtxtx3 = tex_pred_bx3xfxtt.view(B, 3, F, T, T).permute(0, 2, 3, 4, 1)
        textures_nmr = tex_pred_bxfxtxtx3.unsqueeze(4).expand(-1, -1, -1, -1, T, -1)
        return textures_nmr
