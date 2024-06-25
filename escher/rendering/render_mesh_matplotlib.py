import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import igl


def render_mesh_matplotlib(vertices_npy, faces, sides, fnames, optional_target=None):
    m = vertices_npy
    # visualization
    if optional_target is None:
        m = 0.9 * m / np.max(np.linalg.norm(m, axis=1))
    # normalize mesh : divide by max distance from origin
    plt.clf()
    # set face color
    plt.triplot(m[:, 0], m[:, 1], faces, lw=0.3, color=[0.0, 0.0, 0.0])
    # Six radom colors
    colors = np.array([[1, 0.5, 0.5], [0.0, 0.8, 0.6], [0.0, 0.0, 1.0], [0.4, 0.6, 0.1], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if sides is not None:
        for i, (_, value) in enumerate(sides.items()):
            plt.plot(m[value, 0], m[value, 1], "-", color=colors[i, :], lw=1.5)
            plt.plot(m[value[0], 0], m[value[0], 1], "o", color=[0.0, 0.0, 0.0], markersize=4)

    if optional_target is not None:
        plt.plot(optional_target[:, 0], optional_target[:, 1], "o", color=[0.0, 0.0, 1.0], markersize=4)
    # plt.show()
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    # remove axis from render
    plt.axis("off")
    # remove everything outside the render region

    # restrict the renders to [-1,1] on both axes

    plt.tight_layout(pad=0.05)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    if not isinstance(fnames, list):
        fnames = [fnames]
    for fname in fnames:
        plt.savefig(fname)
    plt.close("all")


def render_points_matplotlib(vertices_npy, fnames):
    plt.clf()
    # set face color
    plt.plot(vertices_npy[:, 0], vertices_npy[:, 1], "-", lw=0.3, color=[0.0, 0.0, 0.0])
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    # remove axis from render
    plt.axis("off")
    plt.tight_layout(pad=0.05)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    if not isinstance(fnames, list):
        fnames = [fnames]
    for fname in fnames:
        plt.savefig(fname)
    plt.close("all")


def render_mesh_matplotlib_from_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    V = torch.from_numpy(data["V"])
    T = data["T"]
    UV = torch.from_numpy(data["UV"])
    texture = torch.from_numpy(data["texture"])
    if texture.shape[0] == 3:
        texture = texture.permute(1, 2, 0).contiguous()
    bdry = data["bdry"]
    constraint_data = data["constraint_data"]
    try:
        square_sides = data["square_sides"]
    except:
        MESH_RESOLUTION = int(sqrt(V.shape[0]))
        print(f"found mesh resolution of {MESH_RESOLUTION}")
        from escher.geometry.get_base_mesh import get_2d_square_mesh
        from escher.geometry.split_square_boundary import split_square_boundary

        points, faces_npy, split = get_2d_square_mesh(MESH_RESOLUTION)

        # =========== init uv ==================================================
        normalized_points = points
        normalized_points = normalized_points - normalized_points.min()
        uv = normalized_points / normalized_points.max()
        uv = torch.from_numpy(uv).unsqueeze(0).to("cuda:0")
        normalized_points = 2 * normalized_points / normalized_points.max() - 1

        # tri = Delaunay(points)
        faces = torch.from_numpy(faces_npy)

        # bdry indices of mesh
        bdry = igl.boundary_loop(faces_npy)

        # split the bdry into 4 sides (left,right,top,down)
        square_sides = split_square_boundary.split_square_boundary(points, bdry)

    A = torch.from_numpy(data["R"])
    V = torch.matmul(V, A.permute(1, 0)).numpy()
    render_mesh_matplotlib(V, T, square_sides, path.replace(".pkl", ".eps"))
