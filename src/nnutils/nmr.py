from __future__ import absolute_import, division, print_function

import torch
import neural_renderer

from ..nnutils import geom_utils


def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    return src

class NMR_custom(neural_renderer.Renderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def render(self, vertices, faces, textures=None, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            if textures is not None:
                textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # lighting
        faces_lighting = neural_renderer.vertices_to_faces(vertices, faces)
        if textures is not None:
            textures = neural_renderer.lighting(
                faces_lighting,
                textures,
                self.light_intensity_ambient,
                self.light_intensity_directional,
                self.light_color_ambient,
                self.light_color_directional,
                self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = neural_renderer.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = neural_renderer.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = neural_renderer.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = neural_renderer.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = neural_renderer.projection(vertices, K, R, t, dist_coeffs, orig_size)

        # rasterization
        faces = neural_renderer.vertices_to_faces(vertices, faces)
        out = neural_renderer.rasterize_rgbad(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color, return_alpha=True, return_depth=True, return_rgb=(textures is not None))
        return out['rgb'], out['depth'], out['alpha']


########################################################################
############## Wrapper torch module for SoftRas Renderer ################
########################################################################
class SoftRas(torch.nn.Module):
    """
    Wrapper for soft rasterizezr
    """
    def __init__(self, img_size=256, perspective=True, **kwargs):
        super(SoftRas, self).__init__()
        import soft_renderer
        self.renderer = soft_renderer.SoftRenderer(image_size=img_size, camera_mode='look_at', perspective=perspective, eye=[0, 0, -2.732], **kwargs)
        self.renderer = self.renderer.cuda()

        self.viewing_angle = 30
        self.perspective = perspective
        self.eye = [0, 0, -2.732]
        self.proj_fn = geom_utils.orthographic_proj_withz
        self.offset_z = 5.

    def ambient_light_only(self):
        # Make light only ambient.
        self.renderer.light_intensity_directional = 0.0
        self.renderer.light_intensity_ambient = 1.0

    def set_bgcolor(self, color):
        self.renderer.renderer.background_color = color

    def project_points(self, verts, cams, depth=False):
        proj = self.proj_fn(verts, cams, offset_z=self.offset_z)

        if depth:
            return proj[:, :, :2], proj[:, :, 2]
        else:
            return proj[:, :, :2]

    def project_points_perspective(self, verts, cams, depth=False):
        verts = self.proj_fn(verts, cams, offset_z=self.offset_z)

        verts = neural_renderer.look_at(verts, self.eye)
        if self.perspective:
            verts = neural_renderer.perspective(verts, angle=self.viewing_angle)

        if depth:
            return verts[:, :, :2], verts[:, :, 2]
        else:
            return verts[:, :, :2]


    def forward(self, vertices, faces, cams, textures=None):
        vertices = self.proj_fn(vertices, cams, offset_z=self.offset_z)

        # Flipping the y-axis here to make it align with the image coordinate system!
        vertices[:, :, 1] *= -1
        if textures is None:
            rgba = self.renderer(vertices, faces)
            return rgba[:,3,:,:]
        else:
            # textures: B,F,T,T,T,3
            b,f,t,_,_,_ = textures.shape
            textures = textures[:,:,:,:,0,:].view(b,f,t*t,3)
            rgba = self.renderer(vertices, faces, textures)
            return rgba[:,:3,:,:]

    def render_depth(self, vertices, faces, cams):
        raise NotImplementedError('Softras doesn\'t implement depth rendering')
        vertices = self.proj_fn(vertices, cams, offset_z=self.offset_z)

        # Flipping the y-axis here to make it align with the image coordinate system!
        vertices[:, :, 1] *= -1
        depth = self.renderer(vertices, faces, mode='depth')
        return depth

    def render_texture_mask(self, vertices, faces, cams, textures):
        vertices = self.proj_fn(vertices, cams, offset_z=self.offset_z)

        # Flipping the y-axis here to make it align with the image coordinate system!
        vertices[:, :, 1] *= -1
        # textures: B,F,T,T,T,3
        b,f,t,_,_,_ = textures.shape
        assert(textures.shape == (b,f,t,t,t,3))
        textures = textures[:,:,:,:,0,:].view(b,f,t*t,3)
        rgba = self.renderer(vertices, faces, textures)
        return rgba[:,:3,:,:], rgba[:,3,:,:]


########################################################################
############## Wrapper torch module for Neural Renderer ################
########################################################################
class NeuralRenderer_pytorch(torch.nn.Module):
    """
    This is the core pytorch function to call.
    Every torch NMR has a chainer NMR.
    Only fwd/bwd once per iteration.
    """
    def __init__(self, img_size=256, perspective=True):
        super(NeuralRenderer_pytorch, self).__init__()
        self.renderer = NMR_custom(image_size=img_size, camera_mode='look_at', perspective=perspective)

        self.renderer.eye = [0, 0, -2.732]
        self.renderer = self.renderer.cuda()

        self.proj_fn = geom_utils.orthographic_proj_withz
        self.offset_z = 5.

    def ambient_light_only(self):
        # Make light only ambient.
        self.renderer.light_intensity_directional = 0.0
        self.renderer.light_intensity_ambient = 1.0

    def set_bgcolor(self, color):
        self.renderer.background_color = color

    def project_points(self, verts, cams, depth=False):
        proj = self.proj_fn(verts, cams, offset_z=self.offset_z)

        if depth:
            return proj[:, :, :2], proj[:, :, 2]
        else:
            return proj[:, :, :2]

    def project_points_perspective(self, verts, cams, depth=False):
        verts = self.proj_fn(verts, cams, offset_z=self.offset_z)

        verts = neural_renderer.look_at(verts, self.renderer.eye)
        if self.renderer.perspective:
            verts = neural_renderer.perspective(verts, angle=self.renderer.viewing_angle)

        if depth:
            return verts[:, :, :2], verts[:, :, 2]
        else:
            return verts[:, :, :2]


    def forward(self, vertices, faces, cams, textures=None):
        vertices = self.proj_fn(vertices, cams, offset_z=self.offset_z)

        # Flipping the y-axis here to make it align with the image coordinate system!
        vertices[:, :, 1] *= -1
        if textures is None:
            masks = self.renderer(vertices, faces, mode='silhouettes')
            return masks
        else:
            imgs, _, _ = self.renderer(vertices, faces, textures)
            return imgs

    def render_depth(self, vertices, faces, cams):
        vertices = self.proj_fn(vertices, cams, offset_z=self.offset_z)

        # Flipping the y-axis here to make it align with the image coordinate system!
        vertices[:, :, 1] *= -1
        depth = self.renderer(vertices, faces, mode='depth')
        return depth

    def render_mask_depth(self, vertices, faces, cams):
        vertices = self.proj_fn(vertices, cams, offset_z=self.offset_z)

        # Flipping the y-axis here to make it align with the image coordinate system!
        vertices[:, :, 1] *= -1
        _, depth, alpha = self.renderer.render(vertices, faces)
        return alpha, depth

    def render_texture_mask(self, vertices, faces, cams, textures):
        vertices = self.proj_fn(vertices, cams, offset_z=self.offset_z)

        # Flipping the y-axis here to make it align with the image coordinate system!
        vertices[:, :, 1] *= -1
        rgb, depth, alpha = self.renderer(vertices, faces, textures=textures)
        return rgb, alpha

