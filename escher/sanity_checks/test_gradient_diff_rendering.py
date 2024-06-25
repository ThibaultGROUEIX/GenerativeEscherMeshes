import os
import time

import igl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import escher.geometry.split_square_boundary as split_square_boundary
import torch
import torchvision
from KKTBuilder import KKTBuilder
from rich.traceback import install
from scipy.spatial import Delaunay
from torchvision import transforms
from tqdm import tqdm

import nvdiffrast.torch as dr

install()

from PIL import Image

import escher.geometry.mesh_utils as mesh_utils
import escher.geometry.renderer as renderer

glctx = dr.RasterizeGLContext()


# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
OUTPUT_DIR = "output"

output_images = []

### generating a 2D mesh of a square
nx, ny = (20, 20)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y)
xv = xv.ravel()
yv = yv.ravel()
points = np.stack((xv, yv), axis=1)
tri = Delaunay(points)

# bdry indices of mesh
bdry = igl.boundary_loop(tri.simplices)

# split the bdry into 4 sides (left,right,top,down)
square_sides = split_square_boundary.split_square_boundary(points, bdry)

# generate nx2 list of edge pairs (i,j)
adjacency_list = igl.adjacency_list(tri.simplices)
edge_pairs = []
for r, i in zip(adjacency_list, range(len(adjacency_list))):
    for j in r:
        if i < j:
            edge_pairs.append((i, j))
edge_pairs = np.asarray(edge_pairs)

# parameters of the mapping, weights on each edge (used in Tutte's embedding)
W = torch.nn.Parameter(torch.ones((edge_pairs.shape[0], 1)))

### prepare the solver that will receive W and return mapped vertices of the mesh
# uncomment if you want to pin the bdry vertices to place
# constraint_data =KKTBuilder.pinned_bdry_constraint_matrix(points,bdry)
# constraints for toric boundary (left connects to right, top to bottom)
constraint_data = KKTBuilder.torus_constraint_matrix(points, square_sides)
# the solver itself
solver = KKTBuilder(edge_pairs, points, constraint_data)

optimizer = torch.optim.SGD([W], lr=1, momentum=0.2)
try:
    os.makedirs(OUTPUT_DIR)
except FileExistsError:
    # directory already exists
    pass

n_steps = 100
pbar = tqdm(total=n_steps, desc="steps", position=0)
start_time = time.time()

for iter in range(n_steps):
    optimizer.zero_grad()
    # weights are positive and smaller than 1
    w = torch.special.expit(W) * 0.95 + 0.05
    # get mapped vertices w.r.t w
    mapped, _ = solver.solve(w)

    # Render the mesh
    vertices = mapped
    # Append a z dimension = 1 to the vertices
    vertices = torch.cat((vertices, torch.ones(vertices.shape[0], 1, device=vertices.device)), axis=-1)
    vertices = vertices / 3.0
    faces = torch.from_numpy(tri.simplices)

    colors = torch.ones_like(vertices)
    colors[:, :] = vertices[:, 1:2]
    colors = (colors + 3) / 6.0  # Roughly normalize to [0, 1]

    mesh = mesh_utils.TorchMesh(vertices.float(), faces, colors)
    mv = torch.eye(4, device=DEVICE)[None, ...]
    proj = torch.eye(4, device=DEVICE)[None, ...]

    rendered_img, rendered_depth = renderer.vertex_color_render(
        mesh=mesh,
        mv=mv,
        proj=proj,
        image_size=(512, 512),
        glctx=glctx,
    )
    img = rendered_img.squeeze()[:, :, :3].cpu().permute(2, 0, 1)

    # Loss
    loss = torch.sum(-((img) ** 2))
    loss.backward()
    optimizer.step()
    m = mapped.detach().numpy()

    # visualization
    plt.clf()
    plt.triplot(m[:, 0], m[:, 1], tri.simplices)
    for s in square_sides.values():
        plt.plot(m[s, 0], m[s, 1], "o")
    # plt.show()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{iter}.png"))

    pil_image = Image.open(os.path.join(OUTPUT_DIR, f"{iter}.png"))
    output_images.append(transforms.PILToTensor()(pil_image).float() / 255.0)

    # update the progress bar with per-iteration information
    pbar.set_postfix({"loss": loss.item()})
    pbar.update(1)

torchvision.utils.save_image(output_images, os.path.join(OUTPUT_DIR, "output.png"))
