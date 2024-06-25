import os

import igl
import matplotlib.pyplot as plt
import numpy as np
import escher.geometry.split_square_boundary as split_square_boundary
import torch
from escher.OTE.core.OTESolver import OTESolver
from escher.OTE.tilings.KleinBottle import KleinBottleConstraints
from escher.OTE.tilings.MobiusStrip import MobiusStripConstraints
from escher.OTE.tilings.OrbifoldIHybrid import OrbifoldIHybridConstraints
from escher.OTE.tilings.Cylinder import CylinderConstraints
from escher.OTE.tilings.OrbifoldIHybrid import OrbifoldIHybridConstraints
from escher.OTE.tilings.OrbifoldII_hexagon import OrbifoldIIHexagonConstraints
from escher.OTE.tilings.Torus_hexagon import TorusHexagonConstraints
from escher.OTE.tilings.Reflect632 import Reflect632Constraints
from scipy.spatial import Delaunay
from escher.OTE.core import GlobalDeformation
from escher.geometry.get_base_mesh import get_2d_square_mesh, get_hexagonal_mesh
from escher.geometry.sanity_checks import check_triangle_orientation

OUTPUT_DIR = "output"
# def get_color_function(colors):
#     def col(phase1,phase2):

#         phase =((phase1) % len(colors) + phase2 % 2)
#         phase = round(phase)%len(colors)
#         return colors[phase]
#     return col

### generating a 2D mesh of a square
# nx, ny = (50, 50)
# x = np.linspace(0, 1, nx)
# y = np.linspace(0, 1, ny)
# xv, yv = np.meshgrid(x, y)
# xv = xv.ravel()
# yv = yv.ravel()
# points = np.stack((xv, yv), axis=1)
# points, faces_npy, _, _ = get_2d_square_mesh(50, num_labels=1)
points, faces_npy, square_sides = get_hexagonal_mesh(vertices_per_edge=50)
faces_split = [faces_npy]

# Toggle this off and it misleadingly works
# faces_npy = Delaunay(points).simplices
faces = torch.from_numpy(faces_npy)
check_triangle_orientation(points, faces)

# bdry indices of mesh
bdry = igl.boundary_loop(faces_npy)

# split the bdry into 4 sides (left,right,top,down)
# square_sides = split_square_boundary.split_square_boundary(points, bdry)

# generate nx2 list of edge pairs (i,j)
adjacency_list = igl.adjacency_list(faces_npy)
edge_pairs = []
for r, i in zip(adjacency_list, range(len(adjacency_list))):
    for j in r:
        if i < j:
            edge_pairs.append((i, j))
edge_pairs = np.asarray(edge_pairs)

# parameters of the mapping, weights on each edge (used in Tutte's embedding)
W = torch.nn.Parameter(torch.randn((edge_pairs.shape[0], 1)))

### prepare the solver that will receive W and return mapped vertices of the mesh
# uncomment if you want to pin the bdry vertices to place
# constraint_data = PinnedBoundaryConstraints(points,bdry)
# constraint_data = TorusConstraints(points, square_sides)
constraint_data = OrbifoldIIHexagonConstraints(points, square_sides)

# the solver itself
solver = OTESolver(edge_pairs, points, constraint_data)

optimizer = torch.optim.SGD([W], lr=1, momentum=0.2)
try:
    os.makedirs(OUTPUT_DIR)
except FileExistsError:
    # directory already exists
    pass
global_map = GlobalDeformation.GlobalDeformation(
    constraint_data.get_horizontal_symmetry_orientation(), random_init=False
)
target = {}
for i in range(0,6,2):
    #modulate radius
    temp = np.linspace(0,np.pi,len(square_sides[0]))
    alpha = 0.2
    TARGET_SIN_FREQ = 3
    MIN_RADIUS = 1.3
    MAX_RADIUS = 1.7
    alpha = (MAX_RADIUS-MIN_RADIUS)/2
    R = MIN_RADIUS+alpha*(np.expand_dims(np.cos(temp*TARGET_SIN_FREQ),axis=1)+1) 
    #decompose to polar coords
    theta = np.arctan2(points[square_sides[i],1],points[square_sides[i],0])
    #final target points
    target[i/2] = np.stack((np.cos(theta) , np.sin(theta)),axis=1)*R
for iter in range(800):
    print(iter)
    optimizer.zero_grad()
    # weights are positive and smaller than 1
    # W = torch.nn.Parameter(torch.randn((edge_pairs.shape[0], 1)))
    w = torch.special.expit(W)
    w_solver_input = w#w * 0.95 + (1 - 0.95) / 2
    # import pickle
    # with open("/home/groueix/GenerativeEscherPatterns/output/failure/results_0.pkl", "rb") as f:
    #     results = pickle.load(f)
    #     w_solver_input = torch.tensor(results["w_solver_input"]).float()
    # get mapped vertices w.r.t w
    mapped, _, _ = solver.solve(w_solver_input)
    mid_point = int(square_sides[0].shape[0] / 2)
    loss = torch.sum((mapped[square_sides[0][mid_point], :] - 40)) * 1000
    # loss += 0.02* torch.sum(W**2)
    # loss = 0
    loss += torch.sum((mapped[edge_pairs[:,0],:] - mapped[edge_pairs[:,1],:])**2)*50
    loss.backward()
    optimizer.step()
    # check_triangle_orientation(mapped, faces)
    global_A = global_map.get_matrix(True, constraint_data.get_global_transformation_type())
    maps = constraint_data.get_tiling(1, vertices=mapped, sides=square_sides)

    # loss = -100 * torch.sum((mapped) ** 2)
    # loss.backward()
    # optimizer.step()

    # visualization
    plt.clf()
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    # colors = colors[: constraint_data.tiling_coloring_number()]
    # col_fun = get_color_function(colors)
    # colors = ["red", "blue", "yellow"]
    if False:
        for map in maps:
            m = map.map(mapped.detach().numpy())
            m = GlobalDeformation.map(torch.tensor(m).float(), global_A).detach().numpy()

            plt.triplot(m[:, 0], m[:, 1], faces_npy, lw=0.1, color=colors[map.orientation_index])
            for s in square_sides.values():
                plt.plot(m[s[1:], 0], m[s[1:], 1], "--", lw=0.1, color="black")
    m = mapped.detach().numpy()
    plt.triplot(m[:, 0], m[:, 1], faces_npy, lw=0.1, color=colors[0])
    if iter%10==0:
        for i in range(3):
            t = target[i]
            plt.plot(t[:,0], t[:,1], "o", color="black")
        m = mapped.detach().numpy()

        # plt.triplot(m[:, 0], m[:, 1], tri.simplices)
        # for i, sides in enumerate(constraint_data.get_boundary()):
        #     for side in sides:
        #         plt.plot(m[side, 0], m[side, 1], "--", lw=1, color=colors[i])
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        # plt.show()
        plt.savefig(os.path.join(OUTPUT_DIR, f"minimal_{iter}.pdf"))
        check_triangle_orientation(mapped, faces)
torch.save(W, os.path.join(OUTPUT_DIR, f"W.pt"))
