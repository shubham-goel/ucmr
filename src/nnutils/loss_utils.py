"""
Loss Utils.
"""

from __future__ import absolute_import, division, print_function

import math

import numpy as np
import torch
from scipy import sparse

from . import geom_utils


def maskiou(mask1, mask2):
    """
    masks: ...,h,w
    returns: ...
    """
    mask1 = mask1>0.5
    mask2 = mask2>0.5
    I = (mask1 & mask2).sum(dim=(-1,-2)).float()
    U = (mask1 | mask2).sum(dim=(-1,-2)).float()
    return I/(U + 1e-4)

def depth_loss_fn(depth_render, depth_pred, mask):
    loss_img = torch.nn.functional.relu(depth_pred-depth_render).pow(2) * mask
    loss = loss_img.view(loss_img.size(0), -1).mean(-1)
    return loss, loss_img

def depth_loss_fn_tanh(depth_render, depth_pred, mask, factor=2):
    loss_img = torch.tanh(factor * torch.nn.functional.relu(depth_pred-depth_render)) * mask
    loss = loss_img.view(loss_img.size(0), -1).mean(-1)
    return loss, loss_img

def depth_loss_fn_vis(depth_render, depth_pred, mask):
    loss = torch.nn.functional.relu(depth_pred-depth_render)* mask
    return loss

def mask_loss_fn(mask_pred, mask_gt, reduction='none', **kwargs):
    loss = torch.nn.functional.mse_loss(mask_pred, mask_gt, reduction='none')
    # loss = loss.view(loss.size(0), -1).mean(-1)
    return loss

def reproject_loss_l2(project_points, grid_points, mask):
    non_mask_points = mask.view(mask.size(0), -1).mean(1)
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, project_points.size(-1))
    loss_img = (mask * project_points - mask * grid_points).pow(2).sum(-1)
    loss = loss_img.view(mask.size(0), -1).mean(1)
    loss = loss / (non_mask_points + 1E-10)
    return loss, loss_img

def score_entropy_loss(score_raw):
    """
    score_raw : B,pp
    returns entropy: scalar + log(num_classes)
    """
    score = torch.nn.functional.softmax(score_raw,dim=-1)
    log_score = torch.nn.functional.log_softmax(score_raw,dim=-1)
    entropy = (score*log_score).sum(-1).mean() + math.log(score_raw.shape[1])
    return entropy

def quat_diversity_loss(quat, eps=1e-6):
    """
    quat      : B,pp,4
    returns
    """
    quat0 = quat[:,:,None,:]
    quat1 = quat[:,None,:,:]
    inter_quats = geom_utils.hamilton_product(quat0, geom_utils.quat_inverse(quat1))
    _, angle = geom_utils.quat2axisangle(inter_quats) # B,pp,pp
    if angle.shape[1] > 1:
        idx = angle.argsort(dim=-1)
        min_angle = torch.gather(angle, -1, idx[:,:,1,None])
        return (1-min_angle/np.pi).mean()
    else:
        return angle.min()*0
    # dist = (quat0*quat1).sum(-1)**2
    # # dist = 1 - (2*dist-1).clamp(-1+eps,1-eps).acos()/np.pi  # B,pp,pp: dist = 1 when quaternions are same
    # eye = torch.eye(dist.shape[1],dtype=torch.uint8, device=dist.device)[None,:,:].expand(dist.shape[0],-1,-1)
    # if dist.shape[1] > 1:
    #     return dist[~eye].mean()
    # else:
    #     return dist.mean()*0

def mask_bi_dt_loss(mask1, mask2, reduction='none', mask1_dt=None, mask2_dt=None):
    """
    mask1: N,h,w
    mask2: N,h,w
    """
    N = mask1.shape[0]
    from ..utils import image as image_utils

    if mask1_dt is None:
        mask1_np = mask1.detach().cpu()
        mask1_dt = np.stack([image_utils.compute_dt(mask1_np[i,:,:]) for i in range(N)], axis=0)
        mask1_dt = torch.as_tensor(mask1_dt,dtype=mask1.dtype,device=mask1.device)

    if mask2_dt is None:
        mask2_np = mask2.detach().cpu()
        mask2_dt = np.stack([image_utils.compute_dt(mask2_np[i,:,:]) for i in range(N)], axis=0)
        mask2_dt = torch.as_tensor(mask2_dt,dtype=mask2.dtype,device=mask2.device)

    mask_loss = (mask1_dt + mask2_dt) * (mask1-mask2)**2
    if reduction=='mean':
        return mask_loss.mean()
    elif reduction=='sum':
        return mask_loss.sum()
    elif reduction=='none':
        return mask_loss
    else:
        raise ValueError(f'Unknown reduction: {reduction}')


def mask_l2_dt_loss(mask1, mask2, reduction='none', mask2_dt=None, **kwargs):
    """
    mask1: N,h,w
    mask2: N,h,w
    loss = mask2_dt * mask1 + (mask1 - mask2)**2
    """
    N = mask1.shape[0]
    from ..utils import image as image_utils

    if mask2_dt is None:
        mask2_np = mask2.detach().cpu()
        mask2_dt = np.stack([image_utils.compute_dt(mask2_np[i,:,:]) for i in range(N)], axis=0)
        mask2_dt = torch.as_tensor(mask2_dt,dtype=mask2.dtype,device=mask2.device)

    mask_loss = mask2_dt * mask1 + (mask1 - mask2)**2
    if reduction=='mean':
        return mask_loss.mean()
    elif reduction=='sum':
        return mask_loss.sum()
    elif reduction=='none':
        return mask_loss
    else:
        raise ValueError(f'Unknown reduction: {reduction}')


def texture_dt_loss(texture_flow, dist_transf, vis_rend=None, cams=None, verts=None, tex_pred=None):
    """
    texture_flow: B x ... x 2
    (In normalized coordinate [-1, 1])
    dist_transf: B x 1 x N x N

    Similar to geom_utils.sample_textures
    But instead of sampling image, it samples dt values.
    """
    # Reshape into B x 1 x ... x 2
    b = texture_flow.size(0)
    flow_grid = texture_flow.view(b, 1, -1, 2)
    # B x 1 x 1 x ...
    dist_transf = torch.nn.functional.grid_sample(dist_transf, flow_grid)

    if vis_rend is not None:
        raise NotImplementedError # not implemented for general input shape
        # Visualize the error!
        # B x 3 x F x T*T
        dts = dist_transf.repeat(1, 3, 1, 1)
        # B x 3 x F x T x T
        dts = dts.view(-1, 3, F, T, T)
        # B x F x T x T x 3
        dts = dts.permute(0, 2, 3, 4, 1)
        dts = dts.unsqueeze(4).repeat(1, 1, 1, 1, T, 1) / dts.max()

        from ..utils import bird_vis
        for i in range(dist_transf.size(0)):
            rend_dt = vis_rend(verts[i], cams[i], dts[i])
            rend_img = bird_vis.tensor2im(tex_pred[i].data)
            import matplotlib.pyplot as plt
            plt.ion()
            fig=plt.figure(1)
            plt.clf()
            ax = fig.add_subplot(121)
            ax.imshow(rend_dt)
            ax = fig.add_subplot(122)
            ax.imshow(rend_img)
            import ipdb; ipdb.set_trace()

    return dist_transf.mean()


def texture_loss(img_pred, img_gt, mask_pred, mask_gt):
    """
    Input:
      img_pred, img_gt: B x 3 x H x W
      mask_pred, mask_gt: B x H x W
    """
    mask_pred = mask_pred.unsqueeze(1)
    mask_gt = mask_gt.unsqueeze(1)

    # masked_rend = (img_pred * mask)[0].data.cpu().numpy()
    # masked_gt = (img_gt * mask)[0].data.cpu().numpy()
    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.figure(1)
    # plt.clf()
    # fig = plt.figure(1)
    # ax = fig.add_subplot(121)
    # ax.imshow(np.transpose(masked_rend, (1, 2, 0)))
    # ax = fig.add_subplot(122)
    # ax.imshow(np.transpose(masked_gt, (1, 2, 0)))
    # import ipdb; ipdb.set_trace()

    return torch.nn.L1Loss()(img_pred * mask_pred, img_gt * mask_gt)


def camera_loss(cam_pred, cam_gt, margin, reduce='mean'):
    """
    cam_* are B x 7, [sc, tx, ty, quat]
    Losses are in similar magnitude so one margin is ok.
    """
    rot_pred = cam_pred[:, -4:]
    rot_gt = cam_gt[:, -4:]

    rot_loss = hinge_loss(quat_loss_geodesic(rot_pred, rot_gt), margin).squeeze(-1)
    # Scale and trans.
    st_loss = (cam_pred[:, :3] - cam_gt[:, :3])**2
    st_loss = hinge_loss(st_loss, margin).mean(-1)

    if reduce=='mean':
        return rot_loss.mean() + st_loss.mean()
    else:
        return rot_loss + st_loss

def hinge_loss(loss, margin):
    # Only penalize if loss > margin
    return torch.nn.functional.relu(loss - margin)


def quat_loss_geodesic(q1, q2):
    '''
    Geodesic rotation loss.

    Args:
        q1: N X 4
        q2: N X 4
    Returns:
        loss : N x 1
    '''
    q1 = torch.unsqueeze(q1, 1)
    q2 = torch.unsqueeze(q2, 1)
    q2_conj = torch.cat([ q2[:, :, [0]] , -1*q2[:, :, 1:4] ], dim=-1)
    q_rel = geom_utils.hamilton_product(q1, q2_conj)
    q_loss = 1 - torch.abs(q_rel[:, :, 0])
    # we can also return q_loss*q_loss
    return q_loss


def quat_loss(q1, q2):
    '''
    Anti-podal squared L2 loss.

    Args:
        q1: N X 4
        q2: N X 4
    Returns:
        loss : N x 1
    '''
    q_diff_loss = (q1-q2).pow(2).sum(1)
    q_sum_loss = (q1+q2).pow(2).sum(1)
    q_loss, _ = torch.stack((q_diff_loss, q_sum_loss), dim=1).min(1)
    return q_loss


def triangle_loss(verts, edge2verts):
    """
    Encourages dihedral angle to be 180 degrees.

    Args:
        verts: B X N X 3
        edge2verts: B X E X 4
    Returns:
        loss : scalar
    """
    indices_repeat = torch.stack([edge2verts, edge2verts, edge2verts], dim=2) # B X E X 3 X 4

    verts_A = torch.gather(verts, 1, indices_repeat[:, :, :, 0])
    verts_B = torch.gather(verts, 1, indices_repeat[:, :, :, 1])
    verts_C = torch.gather(verts, 1, indices_repeat[:, :, :, 2])
    verts_D = torch.gather(verts, 1, indices_repeat[:, :, :, 3])

    # n1 = cross(ad, ab)
    # n2 = cross(ab, ac)
    n1 = geom_utils.cross_product(verts_D - verts_A, verts_B - verts_A)
    n2 = geom_utils.cross_product(verts_B - verts_A, verts_C - verts_A)

    n1 = torch.nn.functional.normalize(n1, dim=2)
    n2 = torch.nn.functional.normalize(n2, dim=2)

    dot_p = (n1 * n2).sum(2)
    loss = ((1 - dot_p)**2).mean()
    return loss


def deform_l2reg(V):
    """
    l2 norm on V = B x N x 3
    """
    V = V.view(-1, V.size(2))
    return torch.mean(torch.norm(V, p=2, dim=1))


def entropy_loss(A):
    """
    Input is K x N
    Each column is a prob of vertices being the one for k-th keypoint.
    We want this to be sparse = low entropy.
    """
    entropy = -torch.sum(A * torch.log(A), 1)
    # Return avg entropy over
    return torch.mean(entropy)


def kp_l2_loss(kp_pred, kp_gt):
    """
    L2 loss between visible keypoints.

    \Sum_i [0.5 * vis[i] * (kp_gt[i] - kp_pred[i])^2] / (|vis|)
    """
    criterion = torch.nn.MSELoss()

    vis = (kp_gt[:, :, 2, None] > 0).float()

    # This always has to be (output, target), not (target, output)
    return criterion(vis * kp_pred, vis * kp_gt[:, :, :2])


def lsgan_loss(score_real, score_fake):
    """
    DELETE ME.
    Label 0=fake, 1=real.
    score_real is B x 1, score for real samples
    score_fake is B x 1, score for fake samples

    Returns loss for discriminator and encoder.
    """

    disc_loss_real = torch.mean((score_real - 1)**2)
    disc_loss_fake = torch.mean((score_fake)**2)
    disc_loss = disc_loss_real + disc_loss_fake

    enc_loss = torch.mean((score_fake - 1)**2)

    return disc_loss, enc_loss


class EdgeLoss(torch.nn.Module):
    """
    Edge length should not diverge from the original edge length.

    On initialization computes the current edge lengths.
    """
    def __init__(self, verts, faces, margin=2, use_l2=False):
        # Input:
        #  verts: N x 3
        #  faces: F x 3
        #  (only using the first 2 columns)
        super().__init__()

        self.use_l2 = use_l2
        self.margin = np.log(margin)

        assert(len(verts.shape)==2)
        assert(len(faces.shape)==2)

        edges0 = faces[:,[0,1]]
        edges1 = faces[:,[1,2]]
        edges2 = faces[:,[2,0]]
        _all_edges = torch.cat([edges0,edges1,edges2], dim=0) # E,2
        self.register_buffer('all_edges', _all_edges)

        v0 = torch.gather(verts, 0, self.all_edges[:,0,None].expand(-1,3))
        v1 = torch.gather(verts, 0, self.all_edges[:,1,None].expand(-1,3))
        edge_lengths = torch.sqrt(((v1 - v0)**2).sum(dim=-1)) # E
        self.register_buffer('log_e0', torch.log(edge_lengths))

    def forward(self, verts):
        e1 = self.compute_edgelength(verts)
        if self.use_l2:
            dist = (torch.log(e1) - self.log_e0)**2
            self.dist = torch.nn.functional.relu(dist - self.margin**2)
        else:
            dist = torch.abs(torch.log(e1) - self.log_e0)
            self.dist = torch.nn.functional.relu(dist - self.margin)
        return self.dist.mean()

    def compute_edgelength(self, V):
        v0 = torch.gather(V, 1, self.all_edges[None,:,0,None].expand(V.shape[0],-1,3))
        v1 = torch.gather(V, 1, self.all_edges[None,:,1,None].expand(V.shape[0],-1,3))
        edge_lengths = torch.sqrt(((v1 - v0)**2).sum(dim=-1)) # E
        return edge_lengths # B x E

class EdgeLoss_simple(torch.nn.Module):
    """
    Regularizes p-norm of edge length
    """
    def __init__(self, faces):
        # Input:
        #  faces: B x F x 3
        super(EdgeLoss_simple, self).__init__()
        self.faces = faces
        edge0 = self.faces[:,:,[0,1]]
        edge1 = self.faces[:,:,[1,2]]
        edge2 = self.faces[:,:,[2,0]]
        edges = torch.cat([edge0, edge1, edge2], dim=1) # contains repeated edges also
        edges_pruned = []
        for bb in range(edges.shape[0]):
            all_edges = edges[bb,:,:]
            all_edges = {((u,v) if (u<v) else (v,u)) for (u,v) in all_edges}
            all_edges = torch.tensor(list(all_edges), dtype=edges.dtype, device=edges.device)
            edges_pruned.append(all_edges)
        self.edges = torch.stack(edges_pruned, dim=0) # B,E,2
        self.edges_rep = torch.stack([self.edges, self.edges, self.edges], dim=2) # B,E,3,2

    def forward(self, vertices):
        # vertices: B,V,3
        v0 = torch.gather(vertices, 1, self.edges_rep[:,:,:,0])
        v1 = torch.gather(vertices, 1, self.edges_rep[:,:,:,1])
        edge_length = torch.norm((v0-v1), dim=2) # B,E
        return (edge_length**2).mean()

class GraphLaplacianLoss(torch.nn.Module):
    """
    Encourages vertices to lie close to mean of neighbours
    """
    def __init__(self, faces, numV):
        # Input:
        #  faces: B x F x 3
        super(GraphLaplacianLoss, self).__init__()
        from ..nnutils.laplacian import GraphLaplacian
        self.laplacian = GraphLaplacian(faces, numV)

    def forward(self, verts):
        Lx = self.laplacian(verts) # B,V
        return Lx.mean()

class LaplacianLoss(torch.nn.Module):
    """
    Encourages minimal mean curvature shapes.
    """
    def __init__(self, faces, verts):
        super().__init__()

        # Input:
        #  faces: F x 3
        #  verts: V x 3
        from ..nnutils.laplacian import Laplacian, cotangent

        # V x V
        self.laplacian = Laplacian.apply
        self.cotangent = cotangent

        self.F_np = faces.detach().cpu().numpy()
        self.F = faces.detach()
        self.V = verts.detach()
        self.L = None
        self.Lx = None

        V_np = verts.detach().cpu().numpy()
        batchV = V_np.reshape(-1, 3)
        print('Computing the Laplacian!')
        # Compute cotangents
        C = self.cotangent(self.V[None], self.F[None]).squeeze(0)
        C_np = C.detach().cpu().numpy()
        batchC = C_np.reshape(-1, 3)
        # # Adjust face indices to stack:
        # offset = np.arange(0, verts.size(0)).reshape(-1, 1, 1) * verts.size(1)
        F_np = self.F_np #+ offset
        batchF = F_np.reshape(-1, 3)

        rows = batchF[:, [1, 2, 0]].reshape(-1)
        cols = batchF[:, [2, 0, 1]].reshape(-1)
        # Final size is BN x BN
        BN = batchV.shape[0]
        L = sparse.csr_matrix((batchC.reshape(-1), (rows, cols)), shape=(BN,BN))
        L = L + L.T
        # np.sum on sparse is type 'matrix', so convert to np.array
        M = sparse.diags(np.array(np.sum(L, 1)).reshape(-1), format='csr')
        L = L - M
        # remember this
        self.L = L
        assert(L.shape == (verts.shape[0],verts.shape[0]))

        L_dense = torch.tensor(L.todense(), dtype=verts.dtype)
        self.register_buffer('L_dense', L_dense)

    def __call__(self, verts):
        Lx = torch.matmul(self.L_dense, verts)
        assert(Lx.shape == verts.shape)
        # Lx = torch.sparse.mm(self.L_sparse, batchV).view(verts.shape)

        # Lx.register_hook(lambda x: print('LaplacianLoss:Lx_1', x))
        loss = torch.norm(Lx, p=2, dim=-1).mean()
        # loss.register_hook(lambda x: print('LaplacianLoss:loss', x))
        return loss

    def visualize(self, verts, mv=None):
        # Visualizes the laplacian.
        # Verts is B x N x 3 Variable
        Lx = self.Lx[0].data.cpu().numpy()

        V = verts[0].data.cpu().numpy()

        from psbody.mesh import Mesh
        F = self.laplacian.F_np[0]
        mesh = Mesh(V, F)

        weights = np.linalg.norm(Lx, axis=1)
        mesh.set_vertex_colors_from_weights(weights)

        if mv is not None:
            mv.set_dynamic_meshes([mesh])
        else:
            mesh.show()
            import ipdb; ipdb.set_trace()




class PerceptualTextureLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from ..nnutils.perceptual_loss import PerceptualLoss
        self.add_module('perceptual_loss', PerceptualLoss())

    def forward(self, img_pred, img_gt, mask_pred, mask_gt):
        """
        Input:
          img_pred, img_gt: B x 3 x H x W
        mask_pred, mask_gt: B x H x W
        """
        # mask_pred = mask_pred.unsqueeze(1)
        mask_gt = mask_gt.unsqueeze(1)
        # masked_rend = (img_pred * mask_pred)[0].cpu().numpy()
        # masked_gt = (img_gt * mask_gt)[0].cpu().numpy()
        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.figure(1)
        # plt.clf()
        # fig = plt.figure(1)
        # ax = fig.add_subplot(121)
        # ax.imshow(np.transpose(masked_rend, (1, 2, 0)))
        # ax = fig.add_subplot(122)
        # ax.imshow(np.transpose(masked_gt, (1, 2, 0)))
        # import ipdb; ipdb.set_trace()

        # Only use mask_gt..
        dist = self.perceptual_loss(img_pred * mask_gt, img_gt * mask_gt)
        return dist
