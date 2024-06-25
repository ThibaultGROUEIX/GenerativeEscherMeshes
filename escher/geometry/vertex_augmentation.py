from math import cos, pi, sin
import torch


def vertex_augmentation(mapped, ROTATION_MATRIX, RANDOM_RIGID=True, NO_DEFORMATION=False, IMAGE_BATCH_SIZE=1):
    batch_vertices = []  # will hold a python list of vertices
    for imb in range(IMAGE_BATCH_SIZE):
        # this is the regular code, now in for loop, rendering one image in each iteation
        if NO_DEFORMATION or imb == 0:
            # Keep the first element in the batch without augmentation
            rmat = torch.eye(2)
            delta = torch.Tensor([0, 0])
        else:
            if RANDOM_RIGID:
                theta = torch.randn(1) * pi / 4
                rmat = torch.Tensor([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
                delta = torch.randn([1, 2]) * 0.1
            else:
                rmat = torch.eye(2)
                delta = torch.Tensor([0, 0])
        rmat = torch.matmul(ROTATION_MATRIX, rmat.cuda())
        delta = delta.cuda()
        vertices = torch.matmul(mapped, torch.transpose(rmat, 0, 1)) + delta
        mm = vertices**2
        mm = mm[:, 0] + mm[:, 1]
        mm = torch.sqrt(mm.max())
        vertices = 0.9 * vertices / mm
        # Render the mesh
        # Append a z dimension = 1 to the vertices
        vertices = torch.cat((vertices, torch.ones(vertices.shape[0], 1, device=vertices.device)), axis=-1)
        vertices = vertices
        batch_vertices.append(vertices)
    batch_vertices = torch.stack(batch_vertices)
    return batch_vertices
