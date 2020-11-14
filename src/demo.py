"""
Demo of UCMR. Adapted from CMR

Note that CMR assumes that the object has been detected, so please use a picture of a bird that is centered and well cropped.

Sample usage:

python -m src.demo \
    --pred_pose \
    --pretrained_network_path=cachedir/snapshots/cam/e400_cub_train_cam4/pred_net_600.pth \
    --shape_path=cachedir/template_shape/bird_template.npy
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
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

from .nnutils import predictor as pred_util
from .nnutils import train_utils
from .utils import image as img_util

flags.DEFINE_string('img_path', 'img1.jpg', 'Image to run')
flags.DEFINE_integer('img_size', 256, 'image size the network was trained on.')

opts = flags.FLAGS


def preprocess_image(img_path, img_size=256):
    img = cv2.imread(img_path) / 255.

    # Scale the max image size to be img_size
    scale_factor = float(img_size) / np.max(img.shape[:2])
    img, _ = img_util.resize_img(img, scale_factor)

    # Crop img_size x img_size from the center
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # img center in (x, y)
    center = center[::-1]
    bbox = np.hstack([center - img_size / 2., center + img_size / 2. - 1])

    img = img_util.crop(img, bbox, bgval=1.)

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))

    return img


def visualize(img, outputs, renderer):
    vert = outputs['verts']
    cam = outputs['cam_pred']
    texture = outputs['texture']

    shape_pred = renderer.rgba(vert, cams=cam)[0,:,:,:3]
    img_pred = renderer.rgba(vert, cams=cam, texture=texture)[0,:,:,:3]

    # Different viewpoints.
    vp1 = renderer.diff_vp(
        vert, cam, angle=30, axis=[0, 1, 0], texture=texture, extra_elev=True)[0]
    vp2 = renderer.diff_vp(
        vert, cam, angle=60, axis=[0, 1, 0], texture=texture, extra_elev=True)[0]
    vp3 = renderer.diff_vp(
        vert, cam, angle=60, axis=[1, 0, 0], texture=texture)[0]

    img = np.transpose(img, (1, 2, 0))
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img[:,:,::-1])
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(shape_pred)
    plt.title('pred mesh')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(img_pred[:,:,::-1])
    plt.title('pred mesh w/texture')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(vp1[:,:,::-1])
    plt.title('different viewpoints')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(vp2[:,:,::-1])
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(vp3[:,:,::-1])
    plt.axis('off')
    plt.draw()
    plt.ioff()
    plt.show()
    print('done')

def main(_):

    img = preprocess_image(opts.img_path, img_size=opts.img_size)

    batch = {'img': torch.Tensor(np.expand_dims(img, 0))}

    predictor = pred_util.MeshPredictor(opts)
    outputs = predictor.predict(batch)

    # Texture may have been originally sampled for SoftRas. Resample texture from uv-image for NMR
    outputs['texture'] = predictor.resample_texture_nmr(outputs['uv_image'])

    # This is resolution
    renderer = predictor.vis_rend
    renderer.renderer.renderer.image_size = 512

    visualize(img, outputs, renderer)

if __name__ == '__main__':
    opts.batch_size = 1
    app.run(main)
