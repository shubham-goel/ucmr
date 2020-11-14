from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import torch
from absl import app
from absl import flags

from .nnutils import geom_utils

flags.DEFINE_string('input_file', '', 'Input npz file')
flags.DEFINE_boolean('inputIsCampose', False, 'Input format: campose or raw statistics')
flags.DEFINE_integer('topK', 4, 'Number of camera poses to prune')
flags.DEFINE_boolean('flipZ', True, 'Save z-flipped poses also for a total of 2*topK poses')
flags.DEFINE_string('mergewith', '', 'If all cameras are not present in input_file, merge with this npz file')

def get_campose_dict(input_file, is_campose):
    '''
    Returns campose dict: {frame_id: (cams_px7, scores_p, gt_st_3)}
    '''
    x = np.load(input_file, allow_pickle=True)
    if is_campose:
        campose_dict = x['campose'].item()
    else:
        campose_dict = {}
        gt_cam_nx7 = torch.as_tensor(x['gt_cam'])
        cams_nxpx7 = torch.as_tensor(x['cam_values'])
        score_nxp = torch.as_tensor(x['quat_scores'])
        fids_nx1 = torch.as_tensor(x['frame_id'])

        gt_cam_flip_nx7 = geom_utils.reflect_cam_pose(gt_cam_nx7)
        cams_flip_nxpx7 = geom_utils.reflect_cam_pose(cams_nxpx7)
        fids_flip_nx1 = 1000000 - fids_nx1

        flip = fids_nx1>1000000/2
        gt_cam_nx7 = torch.where(flip, gt_cam_flip_nx7, gt_cam_nx7)
        cams_nxpx7 = torch.where(flip[:,:,None], cams_flip_nxpx7, cams_nxpx7)
        fids_nx1 = torch.where(flip, fids_flip_nx1, fids_nx1)

        assert((fids_nx1>=0).all())
        assert((fids_nx1<1000000/2).all())

        for i in range(fids_nx1.shape[0]):
            fid = int(fids_nx1[i,0])
            gt_st_3 = gt_cam_nx7[i,0:3]
            cams_px7 = cams_nxpx7[i,:]
            score_p = score_nxp[i,:]
            campose_dict[fid] = (cams_px7, score_p, gt_st_3)

    return campose_dict

def trim(opts, inp_campose_dict):
    _factor = 2 if opts.flipZ else 1

    print(f'Selecting top {opts.topK}*{_factor}={opts.topK*_factor} cameras')

    out_campose_dict = {}

    for key in inp_campose_dict:
        cams_px7, score_p, gt_st_3 = inp_campose_dict[key]
        score_k, idx_k = score_p.topk(opts.topK, dim=0)
        cams_kx7 = cams_px7[idx_k, :]

        if opts.flipZ:
            cams_flip_kx7 = geom_utils.reflect_cam_pose_z(cams_kx7)
            cams_kx7 = torch.cat((cams_kx7, cams_flip_kx7), dim=0)
            score_k = torch.cat((score_k, score_k), dim=0)

        out_campose_dict[key] = (cams_kx7, score_k, gt_st_3)

    return out_campose_dict

def main(_):
    opts = flags.FLAGS
    print('Trimming camera poses')
    print(f'input file: {opts.input_file}')
    inp_campose_dict = get_campose_dict(opts.input_file, opts.inputIsCampose)
    print(f'Read inp_campose_dict of size {len(inp_campose_dict)}')

    if opts.mergewith:
        inp_campose_dict2 = get_campose_dict(opts.mergewith, opts.inputIsCampose)
        print(f'Read inp_campose_dict2 of size {len(inp_campose_dict2)}')

        inp_campose_dict = {**inp_campose_dict2, **inp_campose_dict}

    print(f'Trimming campose_dict of size {len(inp_campose_dict)}')
    out_campose_dict = trim(opts, inp_campose_dict)

    output_campose_file = os.path.splitext(opts.input_file)[0] \
                            + f'_PRUNE{opts.topK}' \
                            + (f'_FLIP' if opts.flipZ else '') \
                            + '.npz'
    print(f'output_file: {output_campose_file}')
    print('Saving...')
    np.savez_compressed(output_campose_file, campose=out_campose_dict)
    print('Saved')


if __name__ == "__main__":
    app.run(main)

