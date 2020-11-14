"""
Predicting pose-bin + shape
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision
from absl import app, flags

from ..utils.mesh import (compute_uvsampler_softras,
                          compute_uvsampler_softras_unwrapUV, find_symmetry)
from . import net_blocks as nb
from .geom_utils import (convert_3d_to_uv_coordinates,
                         convert_uv_to_3d_coordinates)
from .mesh_net import (CamPredictor, ConvEncoder, ShapePredictor,
                       TexturePredictorUVShubham)


flags.DEFINE_boolean('pred_shape', True, 'Predict Shape')
flags.DEFINE_boolean('pred_pose', False, 'Predict a single pose')
flags.DEFINE_boolean('pred_texture', True, 'Predict Texture')
flags.DEFINE_boolean('symmetric_mesh', True, 'Mesh symmetry is parametrized')

# Texture Options
flags.DEFINE_boolean('textureImgCustomDimension', False, 'Use custom dimensions for texture image')
flags.DEFINE_integer('textureImgH', 256, 'texture image height')
flags.DEFINE_integer('textureImgW', 256, 'texture image width')
flags.DEFINE_boolean('texture_use_conv_featz', True, 'Use pre-fc conv 4x4 features')
flags.DEFINE_boolean('textureUnwrapUV', False, 'UV map onto 2d image directly, not a sphere')
flags.DEFINE_boolean('texture_uvshift', True, 'Shift uv-map along x to symmetrize it')
flags.DEFINE_boolean('texture_predict_flow', False, 'Predict texture flow, or RGB?')

class ShapeCamTexNet(nn.Module):
    def __init__(self,
            input_shape, nz_feat, perspective,
            symmetric_mesh, pred_shape, pred_pose, pred_texture,
            mean_shape, faces, verts_uv_vx2, faces_uv_fx3x2,
            texture_opts
        ):
        # Input shape is H x W of the image.
        super().__init__()
        self.nz_feat = nz_feat
        self.perspective = perspective
        self.symmetric_mesh = symmetric_mesh
        self.pred_shape = pred_shape
        self.pred_pose = pred_pose
        self.pred_texture = pred_texture
        self.texture_opts = texture_opts

        # Parametrize symmetric mesh
        if self.symmetric_mesh:
            left_sym_idx, right_partner_idx, indep_idx = find_symmetry(mean_shape)
        else:
            left_sym_idx = torch.tensor([], dtype=torch.long)
            right_partner_idx = torch.tensor([], dtype=torch.long)
            indep_idx = torch.arange(mean_shape.shape[0])

        ilr_idx = torch.cat([indep_idx, left_sym_idx, right_partner_idx], dim=0)
        ilr_idx_inv = torch.zeros(mean_shape.shape[0], dtype=torch.long)
        ilr_idx_inv[ilr_idx] = torch.arange(mean_shape.shape[0])
        self.register_buffer('mean_shape_orig', mean_shape+0)
        self.register_buffer('faces', faces+0)
        self.register_buffer('left_sym_idx', left_sym_idx)
        self.register_buffer('right_partner_idx', right_partner_idx)
        self.register_buffer('indep_idx', indep_idx)
        self.register_buffer('ilr_idx_inv', ilr_idx_inv)

        learnable_idx = torch.cat([indep_idx, left_sym_idx])
        assert(learnable_idx.max() < mean_shape.shape[0])
        mean_shape = torch.index_select(mean_shape, 0, learnable_idx)
        self.mean_shape = torch.nn.Parameter(mean_shape)

        assert((self.mean_shape_orig - self.get_mean_shape()).norm(dim=-1).max() <= 1e-4)

        # Encoders
        self.convEncoder = ConvEncoder(input_shape, n_blocks=4)

        # Shape
        if self.pred_shape:
            self.shapePred = nn.Sequential(
                nb.fc_stack(self.convEncoder.out_shape, self.nz_feat, 2),
                ShapePredictor(self.nz_feat, num_verts=mean_shape.shape[0]),
            )
        else:
            self.shapePred = None

        # Pose
        if self.pred_pose:
            self.posePred = nn.Sequential(
                nb.fc_stack(self.convEncoder.out_shape, self.nz_feat, 2),
                CamPredictor(nz_feat=self.nz_feat, scale_bias=(3 if self.perspective else 0.75)),
            )
        else:
            self.posePred = None

        # Texture
        if self.pred_texture:
            sphere_verts_np = convert_uv_to_3d_coordinates(verts_uv_vx2).numpy()
            if not texture_opts.textureUnwrapUV:
                uv_sampler = compute_uvsampler_softras(
                    sphere_verts_np,
                    faces.numpy(),
                    tex_size=texture_opts.tex_size,
                    shift_uv=texture_opts.texture_uvshift
                )
            else:
                faces_uv_fx3x2_np = faces_uv_fx3x2.detach().cpu().numpy()
                uv_sampler = compute_uvsampler_softras_unwrapUV(
                    faces_uv_fx3x2_np,
                    faces.numpy(),
                    tex_size=texture_opts.tex_size,
                    shift_uv=texture_opts.texture_uvshift
                )

            uv_sampler = torch.FloatTensor(uv_sampler)                                      # F' x T x T x 2
            if not texture_opts.textureImgCustomDimension:
                img_H = int(2**np.floor(np.log2(np.sqrt(faces.shape[0]) * texture_opts.tex_size)))
                img_W = 2 * img_H
            else:
                img_H = texture_opts.textureImgH
                img_W = texture_opts.textureImgW
            print(f'textureImg:     {img_H}x{img_W}')
            self.texturePred = TexturePredictorUVShubham(
                    texture_opts.nz_feat, uv_sampler, texture_opts, img_H=img_H, img_W=img_W,
                    predict_flow=texture_opts.texture_predict_flow,
                )
            nb.net_init(self.texturePred)
        else:
            self.texturePred = None


    def forward(self, img):
        """
        img: N,ch,h,w
        returns code: (shape, scale, trans, (quat, score))
        returns uv: N,2,h,w in [-1,1]
        returns mask: N,h,w in [0,1]
        """

        feat = self.convEncoder(img)
        feat_flat = feat.view(feat.shape[0], -1)
        assert(feat_flat.shape[1]==self.convEncoder.out_shape)

        if self.pred_shape:
            shape = self.shapePred(feat_flat)
            shape = self.symmetrize_mesh(shape)
        else:
            shape = torch.zeros((img.shape[0], self.mean_shape_orig.shape[0], 3), dtype=img.dtype, device=img.device)

        if self.pred_pose:
            cam_bx7 = self.posePred(feat_flat)
        else:
            cam_bx7 = None

        if not torch.isfinite(shape).all():
            print('Shape not finite!!')
            import ipdb; ipdb.set_trace()

        if self.pred_texture:
            assert(feat is not None)
            assert(self.texture_opts.texture_use_conv_featz)
            texture = self.texturePred(feat)    # B x F x T x T x 2 or bxvx2
            # else:
            # texture = self.texturePred(feat)    # B x F x T x T x 2 or bxvx2
        else:
            texture = torch.zeros((img.shape[0], self.faces.shape[0], self.texture_opts.tex_size, self.texture_opts.tex_size),
                            dtype=img.dtype, device=img.device)

        return shape, texture, cam_bx7

    def get_mean_shape(self):
        # Add 0 that doesn't pass parameter by reference
        return self.symmetrize_mesh(self.mean_shape)

    def symmetrize_mesh(self, verts):
        """
        Assumes vertices are arranged as [indep, left]
        """
        num_indep = self.indep_idx.shape[0]
        indep = verts[...,:num_indep,:]
        left = verts[...,num_indep:,:]
        right = verts[...,num_indep:,:] * torch.tensor([-1,1,1],dtype=verts.dtype,device=verts.device).view((1,)*(verts.dim()-1)+(3,))
        ilr = torch.cat([indep, left, right], dim=-2)
        assert(self.ilr_idx_inv.max() < ilr.shape[-2]), f'idx ({self.ilr_idx_inv.max()}) >= dim{-2} of {ilr.shape}'
        verts_full = torch.index_select(ilr, -2, self.ilr_idx_inv)
        return verts_full

if __name__ == "__main__":
    raise NotImplementedError
