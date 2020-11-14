"""
Computes Lx and it's derivative, where L is the graph laplacian on the mesh with cotangent weights.

1. Given V, F, computes the cotangent matrix (for each face, computes the angles) in pytorch.
2. Then it's taken to NP and sparse L is constructed.

Mesh laplacian computation follows Alec Jacobson's gptoolbox.
"""

from __future__ import absolute_import, division, print_function

import torch


#############
### Utils ###
#############
def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    return src

class GraphLaplacian(torch.nn.Module):
    def __init__(self, faces, numV):
        super(GraphLaplacian, self).__init__()
        # Faces: M,3
        assert(len(faces.shape)==2)
        assert(faces.shape[1]==3)
        edge0 = faces[:,[0,1]]
        edge1 = faces[:,[1,2]]
        edge2 = faces[:,[2,0]]
        edges = torch.cat([edge0, edge1, edge2], dim=0).cpu() # contains repeated edges also
        laplacian = torch.zeros((numV,numV),dtype=torch.float32)

        edges = {((u,v) if (u<v) else (v,u)) for (u,v) in edges}
        for (u,v) in edges:
            laplacian[u,u] += 1
            laplacian[v,v] += 1
            laplacian[u,v] -= 1
            laplacian[v,u] -= 1
        laplacian = laplacian.to(faces.device)    # V,V
        edges = torch.tensor(list(edges), dtype=faces.dtype, device=faces.device) # E,2

        # Register buffers
        self.register_buffer('laplacian', laplacian)
        self.register_buffer('edges', edges)
        self.register_buffer('faces', faces)

    def forward(self, verts):
        # verts: B,V,3
        lap = self.laplacian[None,:,:,None].expand(verts.shape[0],-1,-1,-1) # B,V,V,1
        verts = verts[:,None,:,:] # B,1,V,3
        diff = (lap*verts).sum(dim=2) # B,V,3
        diff = torch.norm(diff, dim=-1) # B,V
        return diff

########################################################################
################# Wrapper class for a  PythonOp ########################
##### All functions must only use torch Tensors as inputs/outputs ######
########################################################################
class Laplacian(torch.autograd.Function):

    @staticmethod
    def forward(self, V, L):
        # If forward is explicitly called, V is still a Parameter or Variable
        # But if called through __call__ it's a tensor.
        # This assumes __call__ was used.
        #
        # Input:
        #   V: B x N x 3
        #   F: B x F x 3
        # Outputs: Lx B x N x 3
        #
        # Numpy also doesnt support sparse tensor, so stack along the batch
        self.L = L
        V_np = V.cpu().numpy()
        batchV = V_np.reshape(-1, 3)

        Lx = self.L.dot(batchV).reshape(V_np.shape)

        return convert_as(torch.Tensor(Lx), V)

    @staticmethod
    def backward(self, grad_out):
        """
        Just L'g = Lg
        Args:
           grad_out: B x N x 3
        Returns:
           grad_vertices: B x N x 3
        """
        g_o = grad_out.cpu().numpy()
        # Stack
        g_o = g_o.reshape(-1, 3)
        Lg = self.L.dot(g_o).reshape(grad_out.shape)
        ret = convert_as(torch.Tensor(Lg), grad_out)
        return ret, None


def cotangent(V, F):
    # Input:
    #   V: B x N x 3
    #   F: B x F  x3
    # Outputs:
    #   C: B x F x 3 list of cotangents corresponding
    #     angles for triangles, columns correspond to edges 23,31,12

    # B x F x 3 x 3
    indices_repeat = torch.stack([F, F, F], dim=2)

    v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0])
    v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1])
    v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2])

    l1 = torch.sqrt(((v2 - v3)**2).sum(2))
    l2 = torch.sqrt(((v3 - v1)**2).sum(2))
    l3 = torch.sqrt(((v1 - v2)**2).sum(2))

    # semiperimieters
    sp = (l1 + l2 + l3) * 0.5

    # Heron's formula for area
    A = 2*torch.sqrt( sp * (sp-l1)*(sp-l2)*(sp-l3))

    cot23 = (l2**2 + l3**2 - l1**2)
    cot31 = (l1**2 + l3**2 - l2**2)
    cot12 = (l1**2 + l2**2 - l3**2)

    # 2 in batch
    C = torch.stack([cot23, cot31, cot12], 2) / torch.unsqueeze(A, 2) / 4

    return C

def test():
    from ..utils import mesh
    vertices, faces = mesh.create_sphere()
    faces = faces[None, :, :]
    vertices = vertices[None, :, :]
    laplacian = Laplacian(torch.tensor(faces).long())


    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    inp = torch.randn(vertices.shape,dtype=torch.double,requires_grad=True)
    # test = gradcheck(laplacian, inp, eps=1e-3, atol=1e-3)
    # print(test)

    from .loss_utils import LaplacianLoss
    crit = LaplacianLoss(torch.tensor(faces).long())
    test = gradcheck(crit, inp, eps=1e-4, atol=1e-4)
    print(test)

    from tensorboardX import SummaryWriter
    logger = SummaryWriter('temp/')

    vertices = torch.tensor(vertices)
    vertices = vertices + 0.01*vertices.clone().normal_()
    vertices = torch.nn.Parameter(vertices)
    vertices.cuda()
    optimizer = torch.optim.Adam([vertices], lr=0.01)
    for i in range(10):
        loss = crit(vertices)
        weights = crit.visualize(vertices)

        import numpy as np

        # import ipdb; ipdb.set_trace()
        red_color = np.array([255.,0.,0.])
        colors = weights[None,:,None]*red_color[None,None,:]*10
        config_double = {'material':{'cls': 'MeshStandardMaterial', 'side': 2},
                        'lights': [
                                {
                                'cls': 'AmbientLight',
                                'color': '#ffffff',
                                'intensity': 0.75,
                                },
                                # {
                                # 'cls': 'DirectionalLight',
                                # 'color': '#ffffff',
                                # 'intensity': 0.75,
                                # 'position': [0, -1, 2],
                                # }
                                ],}
        logger.add_mesh(f'v{i}', vertices, faces=faces, colors=colors, config_dict=config_double)

        print(f'loss: {loss.item():.6f}; vertices: ({vertices.min():.6f},{vertices.max():.6f})' )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def teapot_smooth_test(loss_lap='loss'):
    from time import time

    # from ..external.neural_renderer import neural_renderer
    # from ..utils.bird_vis import VisRenderer
    import scipy.misc
    import tqdm

    # from psbody.mesh.meshviewer import MeshViewer
    # from psbody.mesh import Mesh

    assert(loss_lap in ['loss', 'lap'])
    print(f'Using {loss_lap}')

    from tensorboardX import SummaryWriter
    logger = SummaryWriter(f'temp/{loss_lap}')

    from ..utils import mesh
    vertices, faces = mesh.create_sphere()
    faces = torch.tensor(faces).long()
    vertices = torch.tensor(vertices)
    # vertices = vertices + 0.05*vertices.clone().normal_()

    # verts = np.tile(verts[None, :, :], (3,1,1))
    # faces = np.tile(faces[None, :, :], (3,1,1))
    vertices = vertices[None, :, :]
    faces = faces[None, :, :]

    vertices = vertices.cuda()
    faces = faces.cuda()

    from .loss_utils import LaplacianLoss
    class SphereModel_Loss(torch.nn.Module):
        def __init__(self):
            super(SphereModel_Loss, self).__init__()
            self.vertices = torch.nn.Parameter(vertices)
            self.loss = LaplacianLoss(faces, verts=vertices)

        def forward(self):
            # import ipdb; ipdb.set_trace()
            return self.loss(self.vertices)
    class SphereModel_Lap(torch.nn.Module):
        def __init__(self):
            super(SphereModel_Lap, self).__init__()
            self.vertices = torch.nn.Parameter(vertices)
            self.laplacian = Laplacian(faces, verts=vertices)

        def forward(self):
            # import ipdb; ipdb.set_trace()
            Lx = self.laplacian(self.vertices)
            return torch.norm(Lx.view(-1, Lx.size(2)), p=2, dim=1).mean()

    opt_model = SphereModel_Lap() if loss_lap=='lap' else SphereModel_Loss()
    optimizer = torch.optim.Adam(opt_model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    print('Smoothing Vertices: ')
    for i in range(200):
        t0 = time()
        optimizer.zero_grad()

        loss = opt_model.forward()

        print(f'loss {loss:.5g}')
        loss.backward()


        import numpy as np

        # import ipdb; ipdb.set_trace()
        red_color = torch.tensor([255.,0.,0.]).double().cuda()
        # weights = torch.norm(Lx, dim=2)
        colors = None #weights[:,:,None]*red_color[None,None,:]*10
        config_double = {'material':{'cls': 'MeshStandardMaterial', 'side': 2},
                        'lights': [
                                {
                                'cls': 'AmbientLight',
                                'color': '#ffffff',
                                'intensity': 0.75,
                                },
                                {
                                'cls': 'DirectionalLight',
                                'color': '#ffffff',
                                'intensity': 0.75,
                                'position': [0, -1, 2],
                                }
                                ],}
        if i%10==0:
            logger.add_mesh(f'v{i}', vertices, faces=faces, colors=colors, config_dict=config_double)

        optimizer.step()


if __name__ == '__main__':
    # test()
    teapot_smooth_test()
