"""
Mesh net model.
"""
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from absl import app, flags

from . import net_blocks as nb
from .spade import SPADEGenerator_noSPADENorm

#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_boolean('symmetric', True, 'Use symmetric mesh or not')
flags.DEFINE_integer('nz_feat', 200, 'Encoded feature size')

flags.DEFINE_boolean('texture', True, 'if true uses texture!')
flags.DEFINE_integer('tex_size', 6, 'Texture resolution per face')

flags.DEFINE_boolean('use_deconv', False, 'If true uses Deconv')
flags.DEFINE_string('upconv_mode', 'bilinear', 'upsample mode')


#------------- Modules ------------#
#----------------------------------#
class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x

class ConvEncoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    """

    def __init__(self, input_shape, n_blocks=4, batch_norm=True):
        super().__init__()
        self.resnet_conv = ResNetConv(n_blocks=4)
        self.enc_conv1 = nb.conv2d(batch_norm, 512, 256, stride=2, kernel_size=4)
        self.out_shape = nc_input = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        nb.net_init(self.enc_conv1)

    def forward(self, img):
        resnet_feat = self.resnet_conv(img)
        out_enc_conv1_bx4x4 = self.enc_conv1(resnet_feat)
        return out_enc_conv1_bx4x4

class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, n_blocks=4, nz_feat=100, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(n_blocks=4)
        self.enc_conv1 = nb.conv2d(batch_norm, 512, 256, stride=2, kernel_size=4)
        nc_input = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2)

        nb.net_init(self.enc_conv1)

    def forward(self, img):
        resnet_feat = self.resnet_conv(img)

        out_enc_conv1_bx4x4 = self.enc_conv1(resnet_feat)
        out_enc_conv1 = out_enc_conv1_bx4x4.view(img.size(0), -1)
        feat = self.enc_fc(out_enc_conv1)

        return feat, out_enc_conv1_bx4x4

class TextureMapPredictor_SPADE_noSPADENorm(nn.Module):
    """
    Outputs UV texture map (no sampling)
    Stores mean paramters, conditions output on input feature using SPADE-normalizations
    """
    def __init__(self, opts, img_H=64, img_W=128, nc_final=3, predict_flow=False, nc_init=256):
        super().__init__()
        self.SPADE_gen = SPADEGenerator_noSPADENorm(opts, img_H, img_W, nc_init, predict_flow=predict_flow, nc_out=nc_final) # nc_init should match value in Encoder()

    def forward(self, conv_feat_bxzxfhxfw):
        self.uvimage_pred = self.SPADE_gen(conv_feat_bxzxfhxfw)
        return self.uvimage_pred

class TexturePredictorUVShubham(nn.Module):
    """
    Outputs mesh texture
    """
    def __init__(self, nz_feat, uv_sampler, opts,
                    img_H=64, img_W=128, n_upconv=5, nc_init=256, predict_flow=False):
        super().__init__()
        self.opts = opts
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init
        self.F = uv_sampler.size(0)
        self.T = uv_sampler.size(1)
        self.predict_flow = predict_flow

        if predict_flow:
            nc_final=2
        else:
            nc_final=3
        self.uvtexture_predictor = TextureMapPredictor_SPADE_noSPADENorm(
            opts, img_H, img_W,
            nc_final=nc_final,
            predict_flow=predict_flow
        )

        assert(uv_sampler.shape == (self.F, self.T, self.T, 2))
        self.register_buffer('uv_sampler', uv_sampler.view(self.F, self.T*self.T, 2))   # F x T x T x 2 --> F x T*T x 2

    def forward(self, feat):
        self.uvimage_pred = self.uvtexture_predictor(feat)

        uv_sampler_batch = self.uv_sampler[None].expand(self.uvimage_pred.shape[0],-1,-1,-1)
        tex_pred = torch.nn.functional.grid_sample(self.uvimage_pred, uv_sampler_batch)
        tex_pred = tex_pred.view(self.uvimage_pred.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1)

        # Contiguous Needed after the permute..
        return tex_pred.contiguous()

class ShapePredictor(nn.Module):
    """
    Outputs mesh deformations
    """
    def __init__(self, nz_feat, num_verts):
        super(ShapePredictor, self).__init__()
        # self.pred_layer = nb.fc(True, nz_feat, num_verts)
        self.pred_layer = nn.Linear(nz_feat, num_verts * 3)

        # Initialize pred_layer weights to be small so initial def aren't so big
        self.pred_layer.weight.data.normal_(0, 0.0001)

    def forward(self, feat):
        delta_v = self.pred_layer(feat)
        delta_v = delta_v.view(delta_v.size(0), -1, 3)
        return delta_v

class QuatPredictor(nn.Module):
    def __init__(self, nz_feat, nz_rot=4):
        super(QuatPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_rot)

    def forward(self, feat):
        quat = self.pred_layer(feat)
        quat = torch.nn.functional.normalize(quat)
        return quat

class ScalePredictor(nn.Module):
    def __init__(self, nz, bias=0.75):
        super(ScalePredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 1)
        self.bias = bias

    def forward(self, feat):
        scale = self.pred_layer(feat) + self.bias  # biasing the scale
        scale = torch.nn.functional.relu(scale) + 1e-12
        # print(self.bias, scale.squeeze(), '\n')
        # print('scale: ( Mean = {}, Var = {} )'.format(scale.mean().item(), scale.var().item()))
        return scale

class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """
    def __init__(self, nz, orth=True):
        super(TransPredictor, self).__init__()
        if orth:
            self.pred_layer = nn.Linear(nz, 2)
        else:
            self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        trans = self.pred_layer(feat)
        # print('trans: ( Mean = {}, Var = {} )'.format(trans.mean().data[0], trans.var().data[0]))
        return trans

class CamPredictor(nn.Module):
    '''
    input feature latent vector feat_bxz
    outputs camera_bx7 [scale, trans_x, trans_y, quat]
    '''
    def __init__(self, nz_feat=100, scale_bias=0.75):
        super().__init__()
        self.quat_predictor = QuatPredictor(nz_feat)
        self.scale_predictor = ScalePredictor(nz_feat, bias=scale_bias)
        self.trans_predictor = TransPredictor(nz_feat)

    def forward(self, feat):
        # shape_pred = self.shape_predictor(feat)
        scale_pred = self.scale_predictor(feat)
        quat_pred = self.quat_predictor(feat)
        trans_pred = self.trans_predictor(feat)

        assert(scale_pred.shape  == (feat.shape[0],1))
        assert(trans_pred.shape  == (feat.shape[0], 2))
        assert(quat_pred.shape  == (feat.shape[0], 4))
        return torch.cat((scale_pred, trans_pred, quat_pred), dim=1)
