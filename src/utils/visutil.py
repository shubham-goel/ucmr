'''Code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix'''
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import tensorboardX
import matplotlib.pyplot as plt
from . import image as image_utils

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy.astype(imtype)


def undo_resnet_preprocess(image_tensor):
    image_tensor = image_tensor.clone()
    image_tensor.narrow(1,0,1).mul_(.229).add_(.485)
    image_tensor.narrow(1,1,1).mul_(.224).add_(.456)
    image_tensor.narrow(1,2,1).mul_(.225).add_(.406)
    return image_tensor


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def visualize_on_sphere(xx,yy,zz,r=1,fig_ax=None,labels=None):
    # Create a sphere
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)

    #Set colours and render
    if fig_ax is None:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig,ax = fig_ax

    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

    if labels is None:
        ax.scatter(xx,yy,zz,color="k",s=20)
    else:
        for i in range(xx.shape[0]):
            ax.scatter(xx[i],yy[i],zz[i],s=30,label=labels[i])


    return fig, ax

def uv2bgr(UV):
    """
    UV: ...,2 in [-1,1]
    returns ...,3
    converts UV values to RGB color
    """
    orig_shape = UV.shape
    UV = UV.reshape(-1,2)
    hue = (UV[:,0]+1)/2 * 179
    light = (UV[:,1]+1)/2
    sat = np.where(light>0.5,(1-light)*2,1) * 255   # [1 -> 1 -> 0]
    val = np.where(light<0.5,light*2,1) * 255       # [0 -> 1 -> 1]
    import cv2
    input_image = np.stack((hue,sat,val),axis=-1)
    output_image = cv2.cvtColor(input_image[None,...].astype(np.uint8), cv2.COLOR_HSV2BGR)
    BGR = output_image.reshape(orig_shape[:-1]+(3,))
    return BGR

def hist2d(xy, **kwargs):
    H, xedges, yedges = np.histogram2d(xy[:,0], xy[:,1], **kwargs)
    return H, xedges, yedges

def hist2d_img(xy, title='', **kwargs):
    H, xedges, yedges = hist2d(xy, **kwargs)
    H = H.T
    fig = plt.figure()
    ax = plt.axes()
    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, H)
    plt.title(title)
    fig.colorbar(im, ax=ax)
    img = tensorboardX.utils.figure_to_image(fig)
    return img

def hist2d_to_img(H, xedges, yedges, title='', **kwargs):
    H = H.T
    fig = plt.figure()
    ax = plt.axes()
    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, H)
    plt.title(title)
    fig.colorbar(im, ax=ax)
    img = tensorboardX.utils.figure_to_image(fig)
    return image_utils.fix_img_dims(img)
