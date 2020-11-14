from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch


def mask_iou(mask1, mask2):
    mask1 = mask1==1
    mask2 = mask2==1
    return (mask1 & mask2).sum()/(mask1 | mask2).sum()

def resize_img(img, scale_factor):
    new_size = (np.round(np.array(img.shape[:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [new_size[0] / float(img.shape[0]),
                     new_size[1] / float(img.shape[1])]
    return new_img, actual_factor


def peturb_bbox(bbox, pf=0, jf=0):
    '''
    Jitters and pads the input bbox.

    Args:
        bbox: Zero-indexed tight bbox.
        pf: padding fraction.
        jf: jittering fraction.
    Returns:
        pet_bbox: Jittered and padded box. Might have -ve or out-of-image coordinates
    '''
    pet_bbox = [coord for coord in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    pet_bbox[0] -= (pf*bwidth) + (1-2*np.random.random())*jf*bwidth
    pet_bbox[1] -= (pf*bheight) + (1-2*np.random.random())*jf*bheight
    pet_bbox[2] += (pf*bwidth) + (1-2*np.random.random())*jf*bwidth
    pet_bbox[3] += (pf*bheight) + (1-2*np.random.random())*jf*bheight

    return pet_bbox


def square_bbox(bbox):
    '''
    Converts a bbox to have a square shape by increasing size along non-max dimension.
    '''
    sq_bbox = [int(round(coord)) for coord in bbox]
    bwidth = sq_bbox[2] - sq_bbox[0] + 1
    bheight = sq_bbox[3] - sq_bbox[1] + 1
    maxdim = float(max(bwidth, bheight))

    dw_b_2 = int(round((maxdim-bwidth)/2.0))
    dh_b_2 = int(round((maxdim-bheight)/2.0))

    sq_bbox[0] -= dw_b_2
    sq_bbox[1] -= dh_b_2
    sq_bbox[2] = sq_bbox[0] + maxdim - 1
    sq_bbox[3] = sq_bbox[1] + maxdim - 1

    return sq_bbox


def crop(img, bbox, bgval=0):
    '''
    Crops a region from the image corresponding to the bbox.
    If some regions specified go outside the image boundaries, the pixel values are set to bgval.

    Args:
        img: image to crop
        bbox: bounding box to crop
        bgval: default background for regions outside image
    '''
    bbox = [int(round(c)) for c in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    im_shape = np.shape(img)
    im_h, im_w = im_shape[0], im_shape[1]

    nc = 1 if len(im_shape) < 3 else im_shape[2]
    img = np.reshape(img, (im_h, im_w, nc))

    img_out = np.ones((bheight, bwidth, nc))*bgval
    x_min_src = max(0, bbox[0])
    x_max_src = min(im_w, bbox[2]+1)
    y_min_src = max(0, bbox[1])
    y_max_src = min(im_h, bbox[3]+1)

    x_min_trg = x_min_src - bbox[0]
    x_max_trg = x_max_src - x_min_src + x_min_trg
    y_min_trg = y_min_src - bbox[1]
    y_max_trg = y_max_src - y_min_src + y_min_trg

    img_out[y_min_trg:y_max_trg, x_min_trg:x_max_trg, :] = img[y_min_src:y_max_src, x_min_src:x_max_src, ...]
    if len(im_shape) < 3:
        img_out = img_out[:,:,0]
    return img_out


def compute_dt(mask):
    """
    Computes distance transform of mask.
    """
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(1-mask) / max(mask.shape)
    return dist


def compute_dt_gpu(mask):
    """
    Computes distance transform of mask (torch tensor).
    """
    # from scipy.ndimage import distance_transform_edt
    # dist0 = distance_transform_edt(1-mask) / max(mask.shape)

    # Runs out of memory
    assert(mask.dim()==2)
    mask_idx = mask.nonzero().float()       # m,2
    ii = torch.arange(mask.shape[0],dtype=mask.dtype,device=mask.device)[:,None].repeat(1,mask.shape[1])
    jj = torch.arange(mask.shape[1],dtype=mask.dtype,device=mask.device)[None,:].repeat(mask.shape[0],1)
    iijj = torch.stack([ii,jj], dim=-1)      # h,w,2
    dist,_ = (iijj[:,:,None,:] - mask_idx[None,None,:,:]).norm(dim=-1).min(dim=-1)
    return dist

def compute_dt_barrier(mask, k=50):
    """
    Computes barrier distance transform of mask.
    """
    from scipy.ndimage import distance_transform_edt
    dist_out = distance_transform_edt(1-mask)
    dist_in = distance_transform_edt(mask)

    dist_diff = (dist_out - dist_in) / max(mask.shape)

    dist = 1. / (1 + np.exp(k * -dist_diff))
    return dist

def fix_img_dims(img):
    if type(img) == torch.Tensor:
        if (img.dim()==3) and (img.shape[0]!=3):
            assert(img.shape[2]==3)
            return img.permute(2,0,1)
        elif img.dim()==2:
            return img[None,:,:].repeat(3,1,1)
        else:
            return img
    elif type(img) == np.ndarray:
        if (len(img.shape)==3) and (img.shape[2] not in [3,4]):
            assert(img.shape[0]in [3,4])
            return img.transpose(1,2,0)
        elif len(img.shape)==2:
            img = np.expand_dims(img, 2)
            return np.tile(img, (1, 1, 3))
        else:
            return img
    elif type(img) == cv2.UMat:
        return img.get()
    else:
        raise TypeError

def concatenate_images1d(img_list, hstack=True):
    """
    img_list is a list of np images.
    concatenate images horizontally if hstack is true, else vertically.
    Pads images with 1 if needed
    """
    img_list = [fix_img_dims(img) for img in img_list]
    max_h = max([img.shape[0] for img in img_list])
    max_w = max([img.shape[1] for img in img_list])
    imgs = []
    for img in img_list:
        if hstack:
            pad_width = ((0,max_h-img.shape[0]),(0,0),(0,0))
        else:
            pad_width = ((0,0),(0,max_w-img.shape[1]),(0,0))
        img = np.pad(img,pad_width,'constant',constant_values=255)
        imgs.append(img)
    if hstack:
        return np.hstack(imgs)
    else:
        return np.vstack(imgs)

def concatenate_images2d(img_list):
    """
    img_list is a possible 2d list of images.
    Concatenate as 2d grid of images
    """
    h_imgs = []
    for h_list in img_list:
        img = concatenate_images1d(h_list, hstack=True)
        h_imgs.append(img)
    img = concatenate_images1d(h_imgs, hstack=False)
    return img

BGR_MEAN=[0.485, 0.456, 0.406]
BGR_STD=[0.229, 0.224, 0.225]
def unnormalize_img(imgs, mean=BGR_MEAN, std=BGR_STD):
    """
    imgs: N,C,h,w
    """
    assert(
        ((len(imgs.shape) == 4) and (imgs.shape[1] == 3)) or
        ((len(imgs.shape) == 3) and (imgs.shape[0] == 3))
    )
    mean = torch.tensor(mean, dtype=torch.float)[:,None,None].to(imgs.device)
    std = torch.tensor(std, dtype=torch.float)[:,None,None].to(imgs.device)
    return imgs*std + mean
