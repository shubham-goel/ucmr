"""
Mesh stuff.
"""

from __future__ import absolute_import, division, print_function

import meshzoo
import numpy as np
import pymesh
import torch

from ..nnutils import geom_utils


def load_obj_file(path, device='cpu'):
    import pymesh
    pmesh = pymesh.load_mesh(path)
    verts = pmesh.vertices
    faces = pmesh.faces
    elem = {
        'verts':verts,
        'faces':faces,
    }
    if pmesh.has_attribute('corner_texture'):
        faces_uv = pmesh.get_face_attribute('corner_texture')
        faces_uv = faces_uv.reshape(faces.shape + (2,))
        elem['faces_uv'] = faces_uv
    return elem

def save_obj_file(path, verts, faces):
    verts = verts.detach().cpu().numpy() if torch.is_tensor(verts) else verts
    faces = faces.detach().cpu().numpy() if torch.is_tensor(faces) else faces

    import pymesh
    pmesh = pymesh.form_mesh(verts, faces)
    pymesh.save_mesh(path, pmesh)

def create_sphere(n_subdivide=3, fix_normals=True):
    # 3 makes 642 verts, 1280 faces,
    # 4 makes 2562 verts, 5120 faces
    verts, faces = meshzoo.iso_sphere(n_subdivide)
    if fix_normals:
        v0 = verts[faces[:,0], :]
        v1 = verts[faces[:,1], :]
        v2 = verts[faces[:,2], :]
        v_m = (v0+v1+v2)/3
        e0 = v1-v0
        e1 = v2-v0
        p = np.cross(e0,e1)
        outwards = (v_m*p).sum(1)
        faces = np.where(outwards[:,None] > 0, faces[:,[0,1,2]], faces[:,[0,2,1]])
    return verts, faces


# def make_symmetric(verts, faces):
#     """
#     Assumes that the input mesh {V,F} is perfectly symmetric
#     Splits the mesh along the X-axis, and reorders the mesh s.t.
#     (so this is reflection on Y-axis..?)
#     [indept verts, right (x>0) verts, left verts]

#     v[:num_indept + num_sym] = A
#     v[:-num_sym] = -A[num_indept:]
#     """
#     left = verts[:, 0] < 0
#     right = verts[:, 0] > 0
#     center = verts[:, 0] == 0

#     left_inds = np.where(left)[0]
#     right_inds = np.where(right)[0]
#     center_inds = np.where(center)[0]

#     num_indept = len(center_inds)
#     num_sym = len(left_inds)
#     assert(len(left_inds) == len(right_inds))

#     # For each right verts, find the corresponding left verts.
#     prop_left_inds = np.hstack([np.where(np.all(verts == np.array([-1, 1, 1]) * verts[ri], 1))[0] for ri in right_inds])
#     assert(prop_left_inds.shape[0] == num_sym)

#     # Make sure right/left order are symmetric.
#     for ind, (ri, li) in enumerate(zip(right_inds, prop_left_inds)):
#         if np.any(verts[ri] != np.array([-1, 1, 1]) * verts[li]):
#             print('bad! %d' % ind)
#             import ipdb; ipdb.set_trace()

#     new_order = np.hstack([center_inds, right_inds, prop_left_inds])
#     # verts i is now vert j
#     ind_perm = np.hstack([np.where(new_order==i)[0] for i in range(verts.shape[0])])

#     new_verts = verts[new_order, :]
#     new_faces0 = ind_perm[faces]

#     new_faces, num_indept_faces, num_sym_faces = make_faces_symmetric(new_verts, new_faces0, num_indept, num_sym)

#     return new_verts, new_faces, num_indept, num_sym, num_indept_faces, num_sym_faces

# def make_faces_symmetric(verts, faces, num_indept_verts, num_sym_verts):
#     """
#     This reorders the faces, such that it has this order:
#       F_indept - independent face ids
#       F_right (x>0)
#       F_left

#     1. For each face, identify whether it's independent or has a symmetric face.

#     A face is independent, if v_i is an independent vertex and if the other two v_j, v_k are the symmetric pairs.
#     Otherwise, there are two kinds of symmetric faces:
#     - v_i is indept, v_j, v_k are not the symmetric paris)
#     - all three have symmetric counter verts.

#     Returns a new set of faces that is in the above order.
#     Also, the symmetric face pairs are reordered so that the vertex order is the same.
#     i.e. verts[f_id] and verts[f_id_sym] is in the same vertex order, except the x coord are flipped
#     """
#     DRAW = False
#     indept_faces = []
#     right_faces = []
#     left_faces = []

#     indept_verts = verts[:num_indept_verts]
#     symmetric_verts = verts[num_indept_verts:]
#     # These are symmetric pairs
#     right_ids = np.arange(num_indept_verts, num_indept_verts+num_sym_verts)
#     left_ids = np.arange(num_indept_verts+num_sym_verts, num_indept_verts+2*num_sym_verts)
#     # Make this for easy lookup
#     # Saves for each vert_id, the symmetric vert_ids
#     v_dict = {}
#     for r_id, l_id in zip(right_ids, left_ids):
#         v_dict[r_id] = l_id
#         v_dict[l_id] = r_id
#     # Return itself for indepentnet.
#     for ind in range(num_indept_verts):
#         v_dict[ind] = ind

#     # Saves faces that contain this verts
#     verts2faces = [np.where((faces == v_id).any(axis=1))[0] for v_id in range(verts.shape[0])]
#     done_face = np.zeros(faces.shape[0])
#     # Make faces symmetric:
#     for f_id in range(faces.shape[0]):
#         if done_face[f_id]:
#             continue
#         v_ids = sorted(faces[f_id])
#         # This is triangles x [x,y,z]
#         vs = verts[v_ids]
#         # Find the corresponding vs?
#         v_sym_ids = sorted([v_dict[v_id] for v_id in v_ids])

#         # Check if it's independent
#         if sorted(v_sym_ids) == sorted(v_ids):
#             # Independent!!
#             indept_faces.append(faces[f_id])
#             # indept_faces.append(f_id)
#             done_face[f_id] = 1
#         else:
#             # Find the face with these verts. (so we can mark it done)
#             possible_faces = np.hstack([verts2faces[v_id] for v_id in v_sym_ids])
#             possible_fids, counts = np.unique(possible_faces, return_counts=True)
#             # The face id is the one that appears 3 times in this list.
#             sym_fid = possible_fids[counts == 3][0]
#             assert(sorted(v_sym_ids) == sorted(faces[sym_fid]))
#             # Make sure that the order of these vertices are the same.
#             # Go in the order of face: f_id
#             face_here = faces[f_id]
#             sym_face_here = [v_dict[v_id] for v_id in face_here]
#             # Above is the same tri as faces[sym_fid], but vertices are in the order of faces[f_id]
#             # Which one is right x > 0?
#             # Only use unique verts in these faces to compute.
#             unique_vids = np.array(v_ids) != np.array(v_sym_ids)
#             if np.all(verts[face_here][unique_vids, 0] < verts[sym_face_here][unique_vids, 0]):
#                 # f_id is left
#                 left_faces.append(face_here)
#                 right_faces.append(sym_face_here)
#             else:
#                 left_faces.append(sym_face_here)
#                 right_faces.append(face_here)
#             done_face[f_id] = 1
#             done_face[sym_fid] = 1
#             # Draw
#             # tri_sym = Mesh(verts[v_sym_ids], [[0, 1, 2]], vc='red')
#             # mv.set_dynamic_meshes([mesh, tri, tri_sym])

#     assert(len(left_faces) + len(right_faces) + len(indept_faces) == faces.shape[0])
#     # Now concatenate them,,
#     new_faces = np.vstack([indept_faces, right_faces, left_faces])
#     # Now sort each row of new_faces to make sure that bary centric coord will be same.
#     num_indept_faces = len(indept_faces)
#     num_sym_faces = len(right_faces)

#     return new_faces, num_indept_faces, num_sym_faces


def compute_edges2verts(verts, faces):
    """
    Returns a list: [A, B, C, D] the 4 vertices for each edge.
    """
    edge_dict = {}
    for face_id, (face) in enumerate(faces):
        for e1, e2, o_id in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
            edge = tuple(sorted((face[e1], face[e2])))
            other_v = face[o_id]
            if edge not in edge_dict.keys():
                edge_dict[edge] = [other_v]
            else:
                if other_v not in edge_dict[edge]:
                    edge_dict[edge].append(other_v)
    result = np.stack([np.hstack((edge, other_vs)) for edge, other_vs in edge_dict.items()])
    return result

def compute_vert2kp(verts, mean_shape):
    # verts: N x 3
    # mean_shape: 3 x K (K=15)
    #
    # computes vert2kp: K x N matrix by picking NN to each point in mean_shape.

    if mean_shape.shape[0] == 3:
        # Make it K x 3
        mean_shape = mean_shape.T
    num_kp = mean_shape.shape[1]

    nn_inds = [np.argmin(np.linalg.norm(verts - pt, axis=1)) for pt in mean_shape]

    dists = np.stack([np.linalg.norm(verts - verts[nn_ind], axis=1) for nn_ind in nn_inds])
    vert2kp = -.5*(dists)/.01
    return vert2kp

def compute_uvsampler(verts_sphere, faces, tex_size=2, shift_uv=False):
    """
    For this mesh, pre-computes the UV coordinates for
    F x T x T points.
    Returns F x T x T x 2
    """
    alpha = np.arange(tex_size, dtype=np.float) / (tex_size-1)
    beta = np.arange(tex_size, dtype=np.float) / (tex_size-1)
    import itertools

    # Barycentric coordinate values
    coords = np.stack([p for p in itertools.product(*[alpha, beta])])
    vs = verts_sphere[faces]
    # Compute alpha, beta (this is the same order as NMR)
    v2 = vs[:, 2]
    v0v2 = vs[:, 0] - vs[:, 2]
    v1v2 = vs[:, 1] - vs[:, 2]
    # F x 3 x T*2
    samples = np.dstack([v0v2, v1v2]).dot(coords.T) + v2.reshape(-1, 3, 1)
    # F x T*2 x 3 points on the sphere
    samples = np.transpose(samples, (0, 2, 1))

    # Now convert these to uv.
    uv = geom_utils.convert_3d_to_uv_coordinates(samples.reshape(-1, 3))
    # uv = uv.reshape(-1, len(coords), 2)

    uv = uv.reshape(-1, tex_size, tex_size, 2)

    if shift_uv:
            # u -> u+0.5
            uv[...,0] = uv[...,0] + 0.5
            uv = np.where(uv>=1, uv-2+1e-12, uv)

    return uv

def compute_uvsampler_softras(verts_sphere, faces, tex_size=2, convert_3d_to_uv=True, shift_uv=False):
    """
    For this mesh, pre-computes the UV coordinates for
    F x T x T points.
    Returns F x T x T x 2
    """
    # alpha_beta_txtx2x2[i,j,0,0] = i + 1/3
    # alpha_beta_txtx2x2[i,j,0,1] = j + 1/3
    # alpha_beta_txtx2x2[i,j,1,0] = i + 2/3
    # alpha_beta_txtx2x2[i,j,1,1] = j + 2/3
    alpha_beta_txtx2x2 = np.zeros((tex_size, tex_size, 2, 2)) # last dim for alpha beta
    alpha_beta_txtx2x2[:,:,:,0] += np.arange(tex_size).reshape(tex_size,1,1)
    alpha_beta_txtx2x2[:,:,:,1] += np.arange(tex_size).reshape(1,tex_size,1)
    alpha_beta_txtx2x2[:,:,0,:] += 1/3
    alpha_beta_txtx2x2[:,:,1,:] += 2/3
    alpha_beta_txtx2x2 = alpha_beta_txtx2x2 / tex_size

    lower_half = alpha_beta_txtx2x2[:,:,0,:]
    # upper_half = np.transpose(alpha_beta_txtx2x2[:,:,1,:], (1,0,2))[::-1,::-1,:]
    upper_half = alpha_beta_txtx2x2[::-1,::-1,1,:]
    upper_half = np.ascontiguousarray(upper_half)
    coords_txtx2 = np.where(lower_half.sum(axis=-1, keepdims=True)<1, lower_half, upper_half)

    # coords_txtx2 = np.transpose(coords_txtx2, (1, 0, 2))
    # coords_txtx2 = np.flip(coords_txtx2,axis=0)
    # coords_txtx2 = np.flip(coords_txtx2,axis=1)
    coords_txtx2 = np.ascontiguousarray(coords_txtx2)

    vs = verts_sphere[faces]
    v0 = vs[:, 2]
    v0v1 = vs[:, 1] - vs[:, 2]
    v0v2 = vs[:, 0] - vs[:, 2]
    samples_Fx3xtxt = np.inner(np.dstack([v0v1, v0v2]), coords_txtx2) + v0.reshape(faces.shape[0], verts_sphere.shape[-1], 1, 1)
    samples_Fxtxtx3 = np.transpose(samples_Fx3xtxt, (0, 2, 3, 1))

    # Now convert these to uv.
    if convert_3d_to_uv:
        uv_Fxtxtx2 =  geom_utils.convert_3d_to_uv_coordinates(samples_Fxtxtx3)

        if shift_uv:
            # u -> u+0.5
            uv_Fxtxtx2[...,0] = uv_Fxtxtx2[...,0] + 0.5
            uv_Fxtxtx2 = np.where(uv_Fxtxtx2>=1, uv_Fxtxtx2-2+1e-12, uv_Fxtxtx2)
        return uv_Fxtxtx2
    else:
        return samples_Fxtxtx3


def compute_uvsampler_unwrapUV(faces_uv, faces, tex_size=2, shift_uv=False):
    """
    For this mesh, pre-computes the UV coordinates for
    F x T x T points.
    Returns F x T x T x 2
    """
    alpha = np.arange(tex_size, dtype=np.float) / (tex_size-1)
    beta = np.arange(tex_size, dtype=np.float) / (tex_size-1)
    import itertools

    # Barycentric coordinate values
    coords = np.stack([p for p in itertools.product(*[alpha, beta])])
    # vs = verts_sphere[faces]

    # Compute alpha, beta (this is the same order as NMR)
    uv2 = faces_uv[:, 2]
    uv02 = faces_uv[:, 0] - faces_uv[:, 2]
    uv12 = faces_uv[:, 1] - faces_uv[:, 2]
    # F x 2 x T*2
    samples = np.dstack([uv02, uv12]).dot(coords.T) + uv2.reshape(-1, 2, 1)
    # F x T*2 x 2 points on the sphere
    samples = np.transpose(samples, (0, 2, 1))

    # # Now convert these to uv.
    # uv = geom_utils.convert_3d_to_uv_coordinates(samples.reshape(-1, 3))
    # # uv = uv.reshape(-1, len(coords), 2)

    uv = samples.reshape(-1, tex_size, tex_size, 2)

    if shift_uv:
            # u -> u+0.5
            uv[...,0] = uv[...,0] + 0.5
            uv = np.where(uv>=1, uv-2+1e-12, uv)

    return uv

def compute_uvsampler_softras_unwrapUV(faces_uv, faces, tex_size=2, shift_uv=False):
    """
    For this mesh, pre-computes the UV coordinates for
    F x T x T points.
    Returns F x T x T x 2
    """
    # alpha_beta_txtx2x2[i,j,0,0] = i + 1/3
    # alpha_beta_txtx2x2[i,j,0,1] = j + 1/3
    # alpha_beta_txtx2x2[i,j,1,0] = i + 2/3
    # alpha_beta_txtx2x2[i,j,1,1] = j + 2/3
    alpha_beta_txtx2x2 = np.zeros((tex_size, tex_size, 2, 2)) # last dim for alpha beta
    alpha_beta_txtx2x2[:,:,:,0] += np.arange(tex_size).reshape(tex_size,1,1)
    alpha_beta_txtx2x2[:,:,:,1] += np.arange(tex_size).reshape(1,tex_size,1)
    alpha_beta_txtx2x2[:,:,0,:] += 1/3
    alpha_beta_txtx2x2[:,:,1,:] += 2/3
    alpha_beta_txtx2x2 = alpha_beta_txtx2x2 / tex_size

    lower_half = alpha_beta_txtx2x2[:,:,0,:]
    # upper_half = np.transpose(alpha_beta_txtx2x2[:,:,1,:], (1,0,2))[::-1,::-1,:]
    upper_half = alpha_beta_txtx2x2[::-1,::-1,1,:]
    upper_half = np.ascontiguousarray(upper_half)
    coords_txtx2 = np.where(lower_half.sum(axis=-1, keepdims=True)<1, lower_half, upper_half)

    # coords_txtx2 = np.transpose(coords_txtx2, (1, 0, 2))
    # coords_txtx2 = np.flip(coords_txtx2,axis=0)
    # coords_txtx2 = np.flip(coords_txtx2,axis=1)
    coords_txtx2 = np.ascontiguousarray(coords_txtx2)

    uv0 = faces_uv[:, 2]
    uv01 = faces_uv[:, 1] - faces_uv[:, 2]
    uv02 = faces_uv[:, 0] - faces_uv[:, 2]
    uv_Fx2xtxt = np.inner(np.dstack([uv01, uv02]), coords_txtx2) + uv0.reshape(faces.shape[0], faces_uv.shape[-1], 1, 1)
    uv_Fxtxtx2 = np.transpose(uv_Fx2xtxt, (0, 2, 3, 1))

    assert(uv_Fxtxtx2.max()<=1+1e-12)
    assert(uv_Fxtxtx2.min()>=-1+1e-12)

    if shift_uv:
        # u -> u+0.5
        uv_Fxtxtx2[...,0] = uv_Fxtxtx2[...,0] + 0.5
        uv_Fxtxtx2 = np.where(uv_Fxtxtx2>=1, uv_Fxtxtx2-2+1e-12, uv_Fxtxtx2)
    return uv_Fxtxtx2


def append_obj(mf_handle, vertices, faces):
    for vx in range(vertices.shape[0]):
        mf_handle.write('v {:f} {:f} {:f}\n'.format(vertices[vx, 0], vertices[vx, 1], vertices[vx, 2]))
    for fx in range(faces.shape[0]):
        mf_handle.write('f {:d} {:d} {:d}\n'.format(faces[fx, 0], faces[fx, 1], faces[fx, 2]))
    return


def find_symmetry(verts):
    """
    verts: N,3
    """
    flip_verts = verts * torch.tensor([[-1,1,1]],dtype=verts.dtype,device=verts.device) # N,3
    # Find kNN btw verts & flip_verts
    verts_dist_all = (verts[:,None,:] - flip_verts[None,:,:]).norm(dim=-1)
    verts_dist_min, min_idx = torch.min(verts_dist_all, dim=1)       # N

    has_sym_mask = (verts_dist_min < 1e-6)      # These have a symmetric partner
    self_mapped_mask = (min_idx==torch.arange(verts.shape[0], device=verts.device))  # These map to themselves
    left_sym_mask = (verts[:,0] < 0) & has_sym_mask & (~self_mapped_mask)   # These need to be learnt, have sym partner
    right_sym_idx = min_idx[left_sym_mask]     # These depend on left_sym_mask
    right_sym_mask = torch.zeros_like(left_sym_mask)
    right_sym_mask[right_sym_idx] = 1
    assert((left_sym_mask & right_sym_mask).sum()==0)
    indep_mask = ~(left_sym_mask | right_sym_mask)  # These need to be learnt, don't have sym partner


    left_sym_idx = left_sym_mask.nonzero().squeeze(1)
    right_partner_idx = min_idx[left_sym_idx]
    indep_idx = indep_mask.nonzero().squeeze(1)

    num_sym = left_sym_idx.shape[0]
    num_indep = indep_idx.shape[0]
    print(f'Mesh contains {num_sym}x2={num_sym*2} symmetric vertices, {num_indep} indep vertices')

    return left_sym_idx, right_partner_idx, indep_idx

def symmetrize(verts, faces, eps=1e-3):
    """
    verts: N,3 (x,y,z) tensor
    faces: F,3 (0,1,2) tensor

    Modifies mesh to make it symmetric about y-z plane
    - Cut mesh into half
    - Copy left half into right half
    - merge, remove duplicate vertices
    """
    # Snap vertices close to centre to centre
    verts_centre_mask = (verts[:,0].abs() < eps)
    verts[verts_centre_mask, 0] = 0

    import pymesh
    pmesh = pymesh.form_mesh(verts.numpy(), faces.numpy())
    pymesh.save_mesh(f'csm_mesh/debug/1.obj', pmesh)

    # Categorize vertices into left (-1), centre(0), right(1)
    verts_side = torch.sign(verts[:,0])

    # Categorize faces into left (-1), centre(0), right(1)
    face_verts_side = torch.index_select(verts_side, 0, faces.view(-1))
    face_verts_side = face_verts_side.contiguous().view(faces.shape[0], 3)
    face_left_mask = (face_verts_side[:,0]==-1) & (face_verts_side[:,1]==-1) & (face_verts_side[:,2]==-1)
    face_right_mask = (face_verts_side[:,0]==1) & (face_verts_side[:,1]==1) & (face_verts_side[:,2]==1)
    face_intesects_yz = (~face_left_mask) & (~face_right_mask)

    # Split intersecting faces
    new_verts = []
    new_faces = []
    for f in face_intesects_yz.nonzero().squeeze(1):
        i0, i1, i2 = faces[f]
        if verts_side[i0]==verts_side[i1]:
            i0, i1, i2 = i2, i0, i1
        elif verts_side[i2]==verts_side[i1]:
            i0, i1, i2 = i0, i1, i2
        elif verts_side[i0]==verts_side[i2]:
            i0, i1, i2 = i1, i0, i2
        elif verts_side[i0]==-1:
            i0, i1, i2 = i0, i1, i2
        elif verts_side[i1]==-1:
            i0, i1, i2 = i1, i0, i2
        elif verts_side[i2]==-1:
            i0, i1, i2 = i2, i0, i1
        else:
            import ipdb; ipdb.set_trace()

        # yz axis intersects i0->i1 & i0->i2
        assert(verts_side[i0] != verts_side[i1])
        assert(verts_side[i0] != verts_side[i2])

        v0 = verts[i0]
        v1 = verts[i1]
        v2 = verts[i2]

        v_n1 = (v0 * v1[0] - v1 * v0[0])/(v1[0] - v0[0])
        v_n2 = (v0 * v2[0] - v2 * v0[0])/(v2[0] - v0[0])

        i_n1 = verts.shape[0] + len(new_verts)
        i_n2 = verts.shape[0] + len(new_verts) + 1
        new_verts.append(v_n1)
        new_verts.append(v_n2)

        new_faces.append((i0, i_n1, i_n2))
        new_faces.append((i1, i_n1, i_n2))
        new_faces.append((i1, i2, i_n2))

    new_verts = torch.stack(new_verts, dim=0)
    new_faces = torch.tensor(new_faces, dtype=faces.dtype, device=faces.device)

    verts = torch.cat([verts, new_verts], dim=0)
    faces = torch.index_select(faces, 0, (~face_intesects_yz).nonzero().squeeze(1))
    faces = torch.cat([faces, new_faces], dim=0)

    import pymesh
    pmesh = pymesh.form_mesh(verts.numpy(), faces.numpy())
    pymesh.save_mesh(f'csm_mesh/debug/2.obj', pmesh)

    # Merge vertices that are very close together
    vertex_mapping = []
    verts_new = []
    for v_id in range(verts.shape[0]):
        if v_id==0:
            vertex_mapping.append(0)
            verts_new.append(verts[0])
            continue
        min_d, min_idx = (verts[0:v_id] - verts[v_id]).norm(dim=-1).min(dim=0)
        if min_d < eps:
            vertex_mapping.append(vertex_mapping[min_idx])
        else:
            vertex_mapping.append(len(verts_new))
            verts_new.append(verts[v_id])
    assert(len(vertex_mapping)==verts.shape[0])
    vertex_mapping = torch.tensor(vertex_mapping, dtype=faces.dtype, device=faces.device)
    verts = torch.stack(verts_new, dim=0)
    faces = vertex_mapping[faces]

    # Remove degenerate faces
    faces_degenerate = (faces[:,0]==faces[:,1]) | (faces[:,0]==faces[:,2]) | (faces[:,2]==faces[:,1])
    faces = torch.index_select(faces, 0, (~faces_degenerate).nonzero().squeeze(1))

    import pymesh
    pmesh = pymesh.form_mesh(verts.numpy(), faces.numpy())
    pymesh.save_mesh(f'csm_mesh/debug/3.obj', pmesh)

    # Delete faces that lie on right side (verts_side==1)
    verts_centre_mask = (verts[:,0].abs() < eps)
    verts[verts_centre_mask, 0] = 0
    verts_side = torch.sign(verts[:,0])
    face_verts_side = torch.index_select(verts_side, 0, faces.view(-1))
    face_verts_side = face_verts_side.contiguous().view(faces.shape[0], 3)
    face_right_mask = (face_verts_side[:,0]==1) | (face_verts_side[:,1]==1) | (face_verts_side[:,2]==1)
    faces = torch.index_select(faces, 0, (~face_right_mask).nonzero().squeeze(1))

    import pymesh
    pmesh = pymesh.form_mesh(verts.numpy(), faces.numpy())
    pymesh.save_mesh(f'csm_mesh/debug/4.obj', pmesh)

    # Flip mesh, merge
    faces_flip = faces + verts.shape[0]
    faces = torch.cat([faces, faces_flip], dim=0)
    verts_flip = verts * torch.tensor([-1,1,1], dtype=verts.dtype, device=verts.device)
    vertex_mapping_flip = torch.arange(verts.shape[0], dtype=faces.dtype, device=faces.device)
    vertex_mapping_flip[~verts_centre_mask] += verts.shape[0]
    vertex_mapping = torch.arange(verts.shape[0], dtype=faces.dtype, device=faces.device)
    vertex_mapping = torch.cat([vertex_mapping, vertex_mapping_flip], dim=0)
    verts = torch.cat([verts, verts_flip], dim=0)
    faces = vertex_mapping[faces]

    import pymesh
    pmesh = pymesh.form_mesh(verts.numpy(), faces.numpy())
    pymesh.save_mesh(f'csm_mesh/debug/5.obj', pmesh)


    pymesh.save_mesh(f'csm_mesh/debug/5.8.obj', pmesh)
    numv = pmesh.num_vertices
    while True:
        pmesh, __ = pymesh.collapse_short_edges(pmesh, rel_threshold=0.4)

        verts = torch.as_tensor(pmesh.vertices, dtype=verts.dtype, device=verts.device)
        faces = torch.as_tensor(pmesh.faces, dtype=faces.dtype, device=faces.device)
        verts_centre_mask = (verts[:,0].abs() < 1e-3)
        verts[verts_centre_mask, 0] = 0

        pmesh = pymesh.form_mesh(verts.numpy(), faces.numpy())
        pmesh, __ = pymesh.remove_isolated_vertices(pmesh)
        pmesh, __ = pymesh.remove_duplicated_vertices(pmesh, tol=eps)
        pmesh, __ = pymesh.remove_duplicated_faces(pmesh)
        pmesh, __ = pymesh.remove_degenerated_triangles(pmesh)
        pmesh, __ = pymesh.remove_isolated_vertices(pmesh)
        # pmesh, __ = pymesh.remove_obtuse_triangles(pmesh, 120.0, 100)
        pmesh, __ = pymesh.collapse_short_edges(pmesh, 1e-2)
        pmesh, __ = pymesh.remove_duplicated_vertices(pmesh, tol=eps)

        if pmesh.num_vertices==numv:
            break
        numv = pmesh.num_vertices
    pymesh.save_mesh(f'csm_mesh/debug/5.9.obj', pmesh)

    verts = torch.as_tensor(pmesh.vertices, dtype=verts.dtype, device=verts.device)
    faces = torch.as_tensor(pmesh.faces, dtype=faces.dtype, device=faces.device)

    # Remove unused vertices
    vertices_used = torch.unique(faces.view(-1))
    vertex_mapping = torch.zeros((verts.shape[0], ), dtype=faces.dtype, device=faces.device)
    vertex_mapping[vertices_used] = torch.arange(vertices_used.shape[0], dtype=faces.dtype, device=faces.device)
    faces = vertex_mapping[faces]
    verts = verts[vertices_used]

    import pymesh
    pmesh = pymesh.form_mesh(verts.numpy(), faces.numpy())
    pymesh.save_mesh(f'csm_mesh/debug/6.obj', pmesh)

    return verts, faces


def fix_mesh(mesh, detail=5e-3):
    # "normal": 5e-3
    # "high":   2.5e-3
    # "low":    2e-2
    # "vlow":   2.5e-2
    bbox_min, bbox_max = mesh.bbox
    diag_len = np.linalg.norm(bbox_max - bbox_min)
    if detail is None:
        detail = 5e-3
    target_len = diag_len * detail
    print("Target resolution: {} mm".format(target_len))

    count = 0
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, __ = pymesh.split_long_edges(mesh, target_len)
    num_vertices = mesh.num_vertices
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-4)
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                preserve_feature=True)

        mesh, __ = pymesh.remove_isolated_vertices(mesh)
        mesh, __ = pymesh.remove_duplicated_vertices(mesh, tol=1e-4)
        mesh, __ = pymesh.remove_duplicated_faces(mesh)
        mesh, __ = pymesh.remove_degenerated_triangles(mesh)
        mesh, __ = pymesh.remove_isolated_vertices(mesh)

        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        print("fix #v: {}".format(num_vertices))
        count += 1
        if count > 10:
            break

    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)

    return mesh

def compute_barycentric_coordinates(vertP, vertA, vertB, vertC):
    AB = vertB - vertA
    AC  = vertC - vertA
    BC  = vertC - vertB

    AP = vertP - vertA
    BP = vertP - vertB
    CP = vertP - vertC

    crossBAC = torch.cross(AB, AC, dim=-1)   # N,3
    crossBAP = torch.cross(AB, AP, dim=-1)   # N,3
    crossCAP = torch.cross(AP, AC, dim=-1)   # N,3
    crossCBP = torch.cross(BC, BP, dim=-1)   # N,3

    areaABC = torch.norm(crossBAC, p=2, dim=-1)                        # N
    nhatABC = torch.nn.functional.normalize(crossBAC, p=2, dim=-1)   # N,3
    areaBAP = (crossBAP*nhatABC).sum(-1)
    areaCAP = (crossCAP*nhatABC).sum(-1)
    areaCBP = (crossCBP*nhatABC).sum(-1)

    w = areaBAP/areaABC
    v = areaCAP/areaABC
    u = areaCBP/areaABC
    barycentric_coordinates = torch.stack([u, v, w], dim=-1)
    barycentric_coordinates = barycentric_coordinates/torch.sum(barycentric_coordinates,dim=-1,keepdim=True)
    return barycentric_coordinates

def puffball(mask):
    """
    input: np.ndarray mask
    Uses puffball approach to inflate mask to 3D shape
    returns a height-map of same size as mask
    """
    mask = mask.astype(np.float)
    mask = np.where(mask>0.5, 1, 0)
    assert(len(mask.shape)==2)

    from scipy.ndimage import distance_transform_edt
    mask_dt = distance_transform_edt(mask)

    cx, cy = mask_dt.nonzero()
    rad = mask_dt[cx, cy]

    x_range = np.arange(mask.shape[0])
    y_range = np.arange(mask.shape[1])
    X,Y = np.meshgrid(x_range, y_range, indexing='ij')

    max_r = rad.max()
    H_softmax = np.zeros_like(mask_dt)
    H_coeff = np.zeros_like(mask_dt)
    for x,y,r in zip(cx,cy,rad):
        h = r**2 - (X-x)**2 - (Y-y)**2
        h = np.where(h<0, 0, h)
        h = np.sqrt(h)
        H_softmax += h * np.exp(h-max_r)
        H_coeff += np.exp(h-max_r)

    return np.where(H_coeff>0, H_softmax/H_coeff, 0)

def depth_map_to_mesh(depth):
    """
    Depth -> Voxels -> Marching Cubes -> Mesh
    """
    assert(np.all(depth>=0))
    max_depth = int(np.ceil(depth.max()))
    voxels = np.zeros((depth.shape[0], depth.shape[1], max_depth))
    for d in range(max_depth):
        vox_slice = np.clip(depth - d, 0, 1)
        voxels[:,:,d] = vox_slice

    # voxels is of size h,w,d. voxels[i,j,k] is far away, more likely to be zero for big k

    # permute voxels s.t. we symmetrize about x axis. depth->x; h->z; w->-y
    voxels = np.transpose(voxels, (2,1,0))[:,::-1,:]
    voxels = np.concatenate((voxels[::-1,:,:], voxels[0:1,:,:], voxels), axis=0)

    from skimage import measure
    verts, faces, normals, values = measure.marching_cubes_lewiner(voxels, 0.5)

    verts = verts-[max_depth,0,0]
    v_max = verts.max()
    v_min = verts.min()
    verts = verts / (v_max - v_min + 1e-6)

    return verts, faces

if __name__ == "__main__":
    # # mask_path =
    # mask = np.zeros((100,100))
    # mask[1:21, 30:70] = 1
    # puff_h = puffball(mask)

    # # import matplotlib.pyplot as plt
    # # plt.imshow(mask)
    # # plt.show()

    # # plt.imshow(puff_h)
    # # plt.show()

    # verts, faces = depth_map_to_mesh(puff_h)
    # import pymesh
    # pmesh = pymesh.form_mesh(verts, faces)
    # pymesh.save_mesh('haha.obj', pmesh)

    vertices = torch.tensor([
        [-1,-1,0.1],
        [1,-1,0.1],
        [-1,1,0.1],
        [1,1,0.1],
    ], dtype=torch.float32)
    faces_all = torch.tensor([
        # [2,1,0],
        [0,1,2],
        # [1,0,2],
        [1,2,0],
        # [2,1,0],
        [2,0,1],
        # [0,2,1],
        [3,1,2],
    ], dtype=torch.long)
    verts_uv = torch.tensor([
        [-1,-1],
        [1,-1],
        [-1,1],
        [1,1],
    ], dtype=torch.float32)

    tex_size = 10



    x = torch.arange(0,1,0.01)[:,None].repeat((1,100))
    y = torch.arange(0,1,0.01)[None,:].repeat((100,1))
    xyz = torch.stack([x,y,torch.zeros_like(x)], dim=-1)
    from . import visutil
    uvimage_pred_hxwx3 = xyz
    uvimage_pred_3xhxw = uvimage_pred_hxwx3.permute((2,0,1))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(uvimage_pred_hxwx3.cpu().numpy())

    for i in [0,1,2]:
        faces = faces_all[[i,3],:]
        faces0 = faces_all[None,0,:]
        uv_sampler_fxtxtx2 = compute_uvsampler_softras(verts_uv.numpy(), faces.numpy(), tex_size=tex_size, convert_3d_to_uv=False)
        uv_sampler_fxtxtx2 = torch.tensor(uv_sampler_fxtxtx2).float()
        uv_sampler_fxttx2 = uv_sampler_fxtxtx2.view(faces.shape[0], tex_size*tex_size, 2)

        # import ipdb; ipdb.set_trace()
        # uv_sampler_fxttx2[:,:,:] = verts_uv[None,None,i,:]

        tex_pred_bx3xfxtt = torch.nn.functional.grid_sample(uvimage_pred_3xhxw[None], uv_sampler_fxttx2[None])
        tex_pred_bxfxtxtx3 = tex_pred_bx3xfxtt.view(1, -1, faces.shape[0], tex_size, tex_size).permute(0, 2, 3, 4, 1)
        tex_pred_bxfxtxtxtx3 = tex_pred_bxfxtxtx3.unsqueeze(4).expand(-1, -1, -1, -1, tex_size, -1)

        from ..nnutils.nmr import SoftRas
        renderer = SoftRas(img_size=256, light_intensity_ambient=1.0, light_intensity_directionals=0.0)
        cams = torch.tensor([3, 0, 0, 1, 0, 0, 0], dtype=torch.float32)        # [sc, tx, ty, quaternions]

        vertices = vertices.cuda()
        faces = faces.cuda()
        cams = cams.cuda()
        tex_pred_bxfxtxtxtx3 = tex_pred_bxfxtxtxtx3.cuda()

        rgb = renderer.forward(vertices[None], faces[None].int(), cams[None], tex_pred_bxfxtxtxtx3)[0,:,:,:]

        plt.figure()
        plt.imshow(rgb.permute((1,2,0)).cpu().numpy())
        plt.title(f'{faces}')

    plt.show()

    import ipdb; ipdb.set_trace()
    while True:
        pass

def fetch_mean_shape(shape_path, mean_centre_vertices=False):
    if shape_path == '':
        print(f'Loading shape from mesh.create_sphere()')
        _v, _f = create_sphere()
        mean_shape = {'verts':_v, 'faces':_f}
    else:
        print(f'Loading shape from {shape_path}')
        extension = shape_path.split('.')[-1]
        if extension in ['npy', 'npz']:
            mean_shape = np.load(shape_path,allow_pickle=True,encoding='latin1').item()
        elif extension in ['obj']:
            mean_shape = load_obj_file(shape_path)
        else:
            raise NotImplementedError
    verts = mean_shape['verts']
    if mean_centre_vertices:
        verts_mean = np.mean(verts,axis=0,keepdims=True)
        verts_mean[0,0] = 0
        print(f'verts:      mean-centering by {verts_mean[0]}')
        verts = verts - verts_mean
    faces = mean_shape['faces']
    try:
        verts_uv = mean_shape['verts_uv']
        print(f'verts_uv:   provided')
    except:
        verts_uv = geom_utils.convert_3d_to_uv_coordinates(verts)
        print(f'verts_uv:   normalizing verts')
    try:
        faces_uv = mean_shape['faces_uv']
        faces_uv = 2*faces_uv-1 # Normalize to [-1,1]
        print(f'faces_uv:   provided')
    except:
        faces_uv = verts_uv[faces]
        print(f'faces_uv:   from verts_uv')

    return {
        'verts': verts,
        'faces': faces,
        'verts_uv': verts_uv,
        'faces_uv': faces_uv,
    }
